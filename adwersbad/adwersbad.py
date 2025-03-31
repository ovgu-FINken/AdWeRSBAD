import json
import math
import random
from collections import defaultdict
from io import BytesIO

# from itertools import pairwise
from typing import List

import numpy as np
import psycopg
import torch
from psycopg import sql
from psycopg.rows import dict_row
from torch.utils.data import IterableDataset

from .class_helpers import create_adwersbad_label_map
from .coders import (
    decode_image,
    decode_image_label,
    decode_lidar,
    decode_lidar_label,
    decode_probabilities,
)
from .config import config
from .db_helpers import (
    get_path_from_to_table,
    get_table_by_name,
    get_table_graph,
    setup_tables,
)
from .log import setup_logger
from .scenarios import adwersbad_scenarios

__all__ = ["Adwersbad"]

log_params = config(section="logging")
logger = setup_logger(__name__, **log_params)


def pairwise(iterable):
    # pairwise('ABCDEFG') → AB BC CD DE EF FG
    iterator = iter(iterable)
    a = next(iterator, None)
    for b in iterator:
        yield a, b
        a = b


def passthrough():
    # defaultdict expects a constructor as argument
    # hence this function returns the identity function
    return lambda x: x


# Dispatch is a mapping from tuple (t, f) to decoding function
# t, f take a value from self.column_types and self.data respectively
# t can be int8, float8, text, bytea, ...
# f can take on image, semantic_labels, points, labels, result, ...
# Compare for loop in prepare_data for usage and context.
# All not explicitly set combinations are handled by pass-through by default

dispatch = defaultdict(passthrough)
# Decode images and corresponding labels
dispatch[("bytea", "image")] = decode_image
dispatch[("bytea", "camera_segmentation")] = decode_image_label
dispatch[("bytea", "prediction")] = decode_probabilities
dispatch[("bytea", "points")] = decode_lidar
dispatch[("bytea", "points_downsampled")] = decode_lidar
dispatch[("bytea", "lidar_segmentation")] = decode_lidar_label
dispatch[("bytea", "lidar_segmentation_downsampled")] = decode_lidar_label
dispatch[("bytea", "result")] = decode_probabilities
dispatch[("bytea", "result_lidar")] = decode_probabilities
dispatch[("bytea", "result_camera")] = decode_probabilities
# dispatch[("text", "lidar_parameters")] = lambda s: json.loads(s)
# dispatch[("text", "camera_parameters")] = lambda s: json.loads(s)


class Adwersbad(IterableDataset):
    """Iterable dataset to load from the adwersbad db"""

    # todo: check for available splits during setup_tables
    AVAILABLE_SPLITS = ("all", "training", "validation", "testing")

    def __init__(
        self,
        data: dict[str, List[str]],
        splits: List[str] = ["all"],
        scenario: str = "all",
        datasets: List[str] = ["all"],
        transforms=None,
        dbtype: str = "psycopg@local",
        itersize=1000,
        shuffle=False,
        offset=0,
        limit=0,
        cursor_type="server",
        orderby=None,
        location=None,
    ):
        """Constructor for the Adwersbad class

        Arguments:
            data List{str} -- database columns to load from the tables (default: {'all'})
            splits {str} -- dataset split {training, validation, testing or all} (default: {'all'})
            scenario {str} -- scenario {all, rain, night, nightrain or duskdawn} to load (default: {'all'})
            dbtype {str} -- database to load from (default: {'psycopg3@local'})
            itersize {int} -- number of rows to load fom the db at once (default: {100})
            offset {int} -- number of rows to skip before returning data from the db
            limit {int} -- maximum number of rows to return from the db, 0 for no limit.
            orderby {str} -- column to order the results by, should be a uniquely identifiable column, such as time or a uid
            location {str} -- location to restrict results to

        """
        super().__init__()
        self.dbparams = config(section=dbtype)
        try:
            self.conn = psycopg.connect(**self.dbparams)
        except psycopg.Error:
            print("Error establishing initial database connection")
            raise
        # setup tables and graph structure
        # todo: field naming consistency, redundant fields
        self.ltables, self.column_to_table_map, self.available_columns = setup_tables(
            self.conn
        )
        self.AVAILABLE_TABLES = [t.name for t in self.ltables]
        self.table_graph = get_table_graph(self.conn, self.ltables)
        logger.info(f"Initiated tables and graph for adwersbad:")
        logger.info(f"Tables: {self.ltables}")
        logger.info(f"Column to table map: {self.column_to_table_map}")
        logger.info(f"Available columns: {self.available_columns}")
        # Test and store input arguments
        # assert all([t in self.AVAILABLE_TABLES for t in tables]), f"invalid table requested: {tables}, available tables are: {self.AVAILABLE_TABLES}"
        assert all(
            [s in self.AVAILABLE_SPLITS for s in splits]
        ), f"invalid split requested: {splits}, available splits are: {self.AVAILABLE_SPLITS}"
        self.splits = splits
        self.fields = data
        self.datasets = datasets
        self.transforms = transforms
        self.offset = offset
        self.limit = limit
        self.orderby = orderby
        self.location = location
        self.data = data
        self.dtypes = {}
        self.field_idxs = {}
        self.dataset_samples = {}
        self.itersize = itersize
        self.cursor_type = cursor_type
        self.shuffle = shuffle
        logger.info(
            f"Adwersbad initialized with {self.splits=}, {self.fields=}, {self.datasets=}"
        )
        assert (
            scenario in adwersbad_scenarios.keys()
        ), f"{scenario} is not a valid scenario, valid scenarios: {[s.name for s in adwersbad_scenarios.values()]}"
        self.scenario = adwersbad_scenarios[scenario]

        for i, field in enumerate(self.fields):
            self.field_idxs.update({field: i})

        self.column_dict, self.scenario_tables = self.setup_columns(self.data)
        try:
            with self.conn.cursor(row_factory=dict_row) as typcursor:
                typcursor.execute("SELECT oid, typname from pg_type order by oid")
                for row in typcursor:
                    self.dtypes.update({row["oid"]: row["typname"]})
            with self.conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    "SELECT DISTINCT dataset, count(dataset) as samples from weather group by dataset"
                )
                for row in cursor:
                    self.dataset_samples.update({row["dataset"]: row["samples"]})

            self.query, self.count_query = self.new_gen_query(
                offset, limit, self.orderby, self.location
            )
            with self.conn.cursor(
                name=f"adwersbad_dtype_cursor", withhold=True
            ) as cursor:
                cursor.itersize = 1
                cursor.execute(self.query)
                self.column_types = [
                    self.dtypes[c.type_code] for c in cursor.description
                ]
            self._count()
            logger.info(f"initialized adwersbad with {self.count} data rows")
            # close connection when setup is done so each copy of the dataset can handle its own connection later
            self.close()
        except psycopg.Error:
            print("Error loading dataset: ")
            raise

    def __iter__(self):
        if self.conn.closed:
            try:
                self.conn = psycopg.connect(**self.dbparams)
            except psycopg.Error:
                print("Error establishing initial database connection")
                raise
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
        else:
            logger.info("Multiple Dataloader workers present, splitting dataset")
            if self.limit > 0 and self.limit * worker_info.num_workers <= self.count:
                per_worker = int(math.ceil(self.limit / worker_info.num_workers))
            else:
                per_worker = int(
                    math.ceil((self.count) / float(worker_info.num_workers))
                )
            if per_worker < 1000 and self.shuffle:
                logger.warning(
                    f"Data shuffling enabled with less than 1000 rows per worker process may result in insufficient shuffling"
                )
            self.worker_id = worker_info.id
            self.offset = self.offset + self.worker_id * per_worker
            self.limit = per_worker
            self.query, self.count_query = self.new_gen_query(
                self.offset, self.limit, self.orderby, self.location
            )
        name = "adwersbad_worker_" + str(self.worker_id)
        logger.info(
            f"Creating new cursor with name {name}, {self.limit=}, {self.offset=} and {self.itersize=}"
        )
        if self.cursor_type == "server":
            with self.conn.cursor(name=name) as cursor:
                cursor.execute(self.query)
                if self.shuffle:
                    rows = list(cursor)
                    random.shuffle(rows)
                    yield from map(self.prepare_data, rows)
                else:
                    yield from map(self.prepare_data, cursor)
        else:
            with self.conn.cursor() as cursor:
                cursor.execute(self.query)
                if self.shuffle:
                    rows = list(cursor)
                    random.shuffle(rows)
                    yield from map(self.prepare_data, rows)
                else:
                    yield from map(self.prepare_data, cursor)
            # self.column_types = [self.dtypes[c.type_code] for c in cursor.description]
            # yield from cursor

    def _count(self):
        logger.info(f"Obtaining rowcount")
        with self.conn.cursor() as cursor:
            cursor.execute(self.count_query)
            self.count = cursor.fetchone()[0]

    def close(self):
        self.conn.close()

    def get_field(self, row: tuple, field: str) -> tuple:
        """Method to get a specific field from a row

        Arguments:
        row {tuple} -- data row obtained from the db
        field {str} -- field to extract from the row

        Returns:
        tuple -- tuple containing the field data
        """
        try:
            idx = self.field_idxs[field]
        except KeyError:
            print("field not found in dataset: " + field)
            raise
        return row[idx]

    def set_field(self, row, field: str, value) -> list:
        """Method to set a specific field in a row

        Arguments:
        row {iterable} -- data row obtained from the db
        field {str} -- field to set
        value -- value to set the field to

        Returns:
        list -- new data row containing the updated field
        """
        row = list(row)  # ensure we are not working on a tuple
        try:
            idx = self.field_idxs[field]
        except KeyError:
            print("field not found in dataset: " + field)
            raise
        row[idx] = value
        return row

    def setup_columns(self, data):
        # ensure input is in correct format:
        # option 1 is to specify only the column names as strings
        f1 = (not isinstance(data, dict)) and all([isinstance(x, str) for x in data])

        # option 2 is a dict[table: [List of columns]]
        f2 = isinstance(data, dict)
        assert isinstance(data, dict)
        # if both are true, something is wrong
        assert not (f1 and f2), f"Invalid data input: {data}"
        if f1:
            assert all(
                [x in self.available_columns for x in data]
            ), f"Invalid data column specified, available data columns: {self.available_columns}"
            used_tables = [self.column_to_table_map[k] for k in self.fields]
            assert all(
                [len(x) == 1 for x in used_tables]
            ), "Non-unique data column specified, please specify columns in (table, column) format"
            scenario_tables = []
            # if we have specific conditions, we need to include the relevant columns
            for k in self.scenario.conditions.keys():
                table = self.column_to_table_map[k][0]
                scenario_tables.append(table)

            # add user-requested columns to our dict
            column_dict = defaultdict(list)
            for c in self.fields:
                table = self.column_to_table_map[c][0]
                column_dict[table].append(c)
            self.data = data
            return column_dict, scenario_tables
        elif f2:
            # TODO: sanity check for dict input
            def flatten(xss):
                return [x for xs in xss for x in xs]

            self.data = flatten(data.values())
            # if we have specific conditions, we need to include the relevant columns
            scenario_tables = []
            for k in self.scenario.conditions.keys():
                # TODO: what if we have scenarios that need data from multiple tables? unlikely, but possible
                table = self.column_to_table_map[k][0]
                scenario_tables.append(table)
            column_dict = defaultdict(list)
            for table, column_list in self.fields.items():
                column_dict[table] += column_list
            if not (self.splits == ["all"]):
                column_dict["weather"] += ["split"]
            if not (self.datasets == ["all"]):
                column_dict["weather"] += ["dataset"]
            return column_dict, scenario_tables
        else:
            raise ValueError(
                "Invalid input columns requested (neither list of strings nor dict)"
            )

    def get_join_path(self, tables_to_join: List[str]) -> List[str]:
        paths = []
        logger.info(f"Creating join path for tables: {tables_to_join}")
        tables_used = [tables_to_join[0]]
        for t1, t2 in pairwise(tables_to_join):
            path = get_path_from_to_table(self.table_graph, t1, t2)
            for source, target, columns in path:
                if not target in tables_used:
                    paths.append((source, target, columns))
                tables_used.append(target)
                logger.info(f"Added {target} table to joins via {path}")
                logger.info(f"Tables to join: {tables_to_join}")
                logger.info(f"Tables alread joined: {tables_used}")
            # check if we already have all tables
            if set(tables_to_join).issubset(tables_used):
                return paths
        return None

    def new_gen_query(self, offset, limit, orderby, location) -> sql.Composed:
        sql_query = sql.Composed("")
        # sanity checks, input format - 1= simple columns, 2= table-column pairs
        logger.info(f"Formatted input columns into {self.column_dict}")

        # To get all our data together, we need to possibly join 2+ tables
        tables_to_join = list(self.column_dict.keys()) + self.scenario_tables
        join_conditions = []
        if len(tables_to_join) == 1:
            # yay, no joins
            join_paths = []
        else:
            # ohno
            # there could probably be a lot of optimization for finding otpimal paths in the table-tree (i.e including tables on the way)
            # but our tree is literally like 6 tables, with degree=1 so..whatever
            join_paths = self.get_join_path(tables_to_join)
            assert (
                join_paths is not None
            ), f"Error fetching join path for tables {tables_to_join}"
        start_table = tables_to_join[0]
        if len(join_paths) > 0:
            # start_table = join_paths[0][0]
            join_conditions = []
            for table1, table2, join_columns in join_paths:
                join_condition = sql.SQL("INNER JOIN {} ON {}").format(
                    sql.Identifier(table2),
                    sql.SQL(" AND ").join(
                        sql.SQL("{}.{} = {}.{}").format(
                            sql.Identifier(table1),
                            sql.Identifier(t1c),
                            sql.Identifier(table2),
                            sql.Identifier(t2c),
                        )
                        for t1c, t2c in join_columns
                    ),
                )
                join_conditions.append(join_condition)

        column_list = []
        for table, columns in self.column_dict.items():
            for col in columns:
                column_list.append((table, col))

        if (
            len(self.splits) == 1
            and self.splits[0] == "all"
            and self.scenario.name == "all"
            and self.datasets[0] == "all"
        ):
            split_condition = sql.SQL("")
        else:
            split_conditions = sql.SQL("")
            tod_conditions = sql.SQL("")
            weather_conditions = sql.SQL("")
            dataset_conditions = []
            scenario_conditions = []
            split = False
            tod = False
            weather = False
            if not (self.splits[0] == "all"):
                # for s in self.splits:
                #     split_conditions.append(sql.SQL("{}").format(
                #                             sql.Literal(s)))
                # split_conditions = sql.SQL(',').join(self.splits)
                split_literals = sql.SQL(", ").join(map(sql.Literal, self.splits))
                split_conditions = sql.SQL("split IN ({})").format(split_literals)
                split = True
            if self.scenario:
                for column, values in self.scenario.conditions.items():
                    # for value in values:
                    if column == "tod":
                        # tod_conditions.append(sql.SQL(" {} = {}").format(
                        #                         sql.Identifier(column),
                        #                         sql.Literal(value)))
                        tod_literals = sql.SQL(", ").join(map(sql.Literal, values))
                        tod_conditions = sql.SQL("tod IN ({})").format(tod_literals)
                        tod = True
                    elif column == "weather":
                        # weather_conditions.append(sql.SQL(" {} = {}").format(
                        #                         sql.Identifier(column),
                        #                         sql.Literal(value)))
                        weather_literals = sql.SQL(", ").join(map(sql.Literal, values))
                        weather_conditions = sql.SQL("weather IN ({})").format(
                            weather_literals
                        )
                        weather = True
            if not (self.datasets[0] == "all"):
                for d in self.datasets:
                    dataset_conditions.append(
                        sql.SQL(" dataset = {}").format(sql.Literal(d))
                    )

            if (weather or tod) and split:
                scenario_and = "AND"
            else:
                scenario_and = ""
            if weather and tod:
                tod_and = "AND"
            else:
                tod_and = ""
            if dataset_conditions and (split or weather or tod):
                dataset_and = "AND"
            else:
                dataset_and = ""
            # if len(split_conditions) > 0 and len(scenario_conditions) >0 and len(dataset_conditions) >0:
            split_condition = sql.SQL("WHERE {} {} {} {} {} {} {}").format(
                # sql.SQL(' OR ').join(split_conditions),
                split_conditions,
                sql.SQL(scenario_and),
                # sql.SQL(' OR ').join(tod_conditions),
                tod_conditions,
                sql.SQL(tod_and),
                # sql.SQL(' OR ').join(weather_conditions),
                weather_conditions,
                sql.SQL(dataset_and),
                sql.SQL(" OR ").join(dataset_conditions),
            )
            # elif len(scenario_conditions) > 0 and len(split_conditions) == 0:
            #     split_condition = sql.SQL("WHERE {}").format(
            #         sql.SQL(' AND ').join(scenario_conditions))
            # elif len(split_conditions) > 0 and len(scenario_conditions) == 0:
            #     split_condition = sql.SQL("WHERE {}").format(
            #         sql.SQL(' OR ').join(split_conditions))
            # else:
            # I dont think this code can be reached
            # raise NotImplementedError("Guess I was wrong")
            # if len(datasets) > 1 or not (datasets[0] == 'all'):
            #     split_condition = (sql.SQL("WHERE {}").format(
            #             sql.SQL(' AND ').join(dataset_conditions))).join(split_condition)
        sql_query = sql.SQL(
            """
            SELECT {}
            FROM {} {} {}
            """
        ).format(
            sql.SQL(", ").join(
                sql.SQL("{}.{}").format(sql.Identifier(table), sql.Identifier(column))
                for table, column in column_list
            ),
            sql.Identifier(start_table),
            sql.SQL(" ").join(join_conditions),
            split_condition,
        )
        # count_query = sql.SQL(
        #     """
        #     SELECT count(*)
        #     FROM {} {} {}
        #     """
        # ).format(
        #     sql.Identifier(start_table),
        #     sql.SQL(" ").join(join_conditions),
        #     split_condition,
        # )
        if location:
            sql_query = sql_query + sql.SQL(" WHERE weather.location={}").format(
                sql.Literal(location)
            )
        if orderby:
            sql_query = sql_query + sql.SQL(" ORDER BY {}").format(
                sql.Identifier(orderby)
            )
        if limit > 0:
            if not orderby:
                # limit needs an orderby to return predictable results
                # any valid selectable column can be used to order
                # in our case that means we can use a uid column that matches the queried table
                # preferably weather_uid, but if only label tables are selected from, we have to use camera_uid or lidar_uid
                # technically we could also just check for selected uid rows first, but this is already kind of a monstrosity so
                # TODO
                logger.debug(
                    f"Limit specified without orderby, searching for suitable uid column for ordering {column_list=}"
                )
                index_column = next(
                    (
                        (table, "weather_uid")
                        for table, _ in column_list
                        if table in {"weather", "camera", "lidar"}
                    ),
                    next(
                        (
                            (
                                (table, "camera_uid")
                                if "camera" in table
                                else (table, "lidar_uid")
                            )
                            for table, _ in column_list
                            if "camera" in table or "lidar" in table
                        ),
                        None,
                    ),
                )
                if index_column:
                    logger.debug(f"Found {index_column=} for ordering")
                else:
                    logger.error(f"No indexable column found for {column_list=}")
                # index_column = next(
                #     (
                #         "weather_uid"
                #         for table, _ in column_list
                #         if table in ["weather", "camera", "lidar"]
                #     ),
                #     None,
                # )
                # if not index_column:
                #     for table, _ in column_list:
                #         if "camera" in table:
                #             index_column = "camera_uid"
                #             break
                #         elif "lidar" in table:
                #             index_column = "lidar_uid"
                #             break
                assert (
                    index_column
                ), "No indexable column found for query, please specify manually"

                sql_query = sql_query + sql.SQL(" ORDER BY {}.{}").format(
                    sql.Identifier(index_column[0]),
                    sql.Identifier(index_column[1]),
                )

            sql_query = sql_query + sql.SQL(" LIMIT {}").format(limit)

        if offset > 0:
            sql_query = sql_query + sql.SQL(" OFFSET {}").format(offset)
        count_query = (
            sql.SQL("SELECT COUNT(*) FROM (") + sql_query + sql.SQL(") AS count")
        )

        logger.info(f"Generated count query: {count_query.as_string(self.conn)}")
        logger.info(f"Generated SQL query: {sql_query.as_string(self.conn)}")
        return sql_query, count_query

    def calculate_stats(self):
        sum = 0
        n = 0
        for x in self:
            sum += x[0]
            n += 1
        if n == 0:
            return 0
        return sum / n

    def prepare_data(self, row: tuple) -> list:
        """Method to extract data as stored (eg. memoryview, bytea) into a more conventional dtype

        For example:
            image: from bytea -> torch.Tensor of dtype float32 and shape [C, H, W] with values in [0, 1]
            label: from bytea -> torch.Tensor of dtype int64 and shape [H, W]
            lidar: from ...

        Arguments:
            row {tuple} -- data (row obtained from the db) to be prepared

        Returns:
            new_data -- list of prepared data
        """
        sample = []
        for d, t, f in zip(row, self.column_types, self.data):
            decode_fn = dispatch[(t, f)]
            sample.append(decode_fn(d))

        if self.transforms is not None:
            sample = self.transforms(*sample)

        return sample

    def flip_aug(self, row) -> list:
        """Method to perform flip augmentation

        Arguments:
            row {iterable} -- data row obtained from the db, this should contain the lidar_points field

        Returns:
            list -- new data row containing the flipped data
        """
        row = list(row)  # ensure we are not working on a tuple
        points = self.get_field(row, "lidar_points")
        flip_type = np.random.choice(4, 1)
        if flip_type == 1:
            points[:, 0] = -points[:, 0]
        elif flip_type == 2:
            points[:, 1] = -points[:, 1]
        elif flip_type == 3:
            points[:, :2] = -points[:, :2]
        row = self.set_field(row, "lidar_points", points)
        return row

    def __new_iter__(self):
        """idea: load next batch in the background after yielding current batch"""
        raise NotImplementedError

    def show_image(self, tensor: torch.Tensor, dummy: bool = False):
        """Method to show images from the current tensor.

        Arguments:
            tensor {torch.Tensor} -- tensor to extract images from
            dummy {bool} -- For testing purposes. If true, will only load images and not display them. (default: {False})

        Returns:
            None
        """
        try:
            img = self.get_field(tensor, "image").numpy()[0]
        except KeyError:
            logger.info("No image loeaded in tensor, aborting show_image")
            return
        try:
            label = self.get_field(tensor, "semantic_labels").numpy()[0]
            has_label = True
        except KeyError:
            has_label = False
            logger.info("No label loaded in tensor, displaying image without labels")
        import matplotlib.pyplot as plt

        if has_label:
            _, axis = plt.subplots(1, 2)
            cmap = create_adwersbad_label_map("id", "color")
            label_colors = np.array([[cmap[id_] for id_ in row] for row in label])
            axis[0].imshow(img)
            axis[1].imshow(label_colors)
        else:
            plt.imshow(img)

        if not dummy:
            plt.show()  # Default is a blocking call
        plt.close("all")

    def show_pointcloud(self, tensor, dummy=False, jupyter=False):
        import open3d as o3

        """Method to show a pointcloud from the current tensor
            CURRENTLY DEFUNCT
        Arguments:
        tensor {list} -- tensor to extract pointcloud from
        dummy {bool} -- whether to only load data and dont show an image, mainly used for profiling (default: {False})
        jupyter {bool} -- whether to show the pointcloud in a jupyter notebook (default: {False})

        Returns:
        pcl -- pointcloud object if jupyter is true, else None
        """
        raise NotImplementedError
        fields = self.data.replace(" ", "").splits(",")
        idx = None
        idx_label = None
        if "lidar_points" in fields:
            idx = fields.index("lidar_points")
        if "lidar_labels" in fields:
            idx_label = fields.index("lidar_labels")
        pcl = o3.geometry.PointCloud()
        pcl.points = o3.utility.Vector3dVector(np.load(BytesIO(tensor[idx][0])))
        if idx_label is not None:
            pcl.colors = o3.utility.Vector3dVector(
                np.load(BytesIO(tensor[idx_label][0]))
            )
        if not dummy:
            if jupyter:
                # we simply return the pointcloud for jupyter to draw
                return pcl
            else:
                o3.visualization.draw_geometries([pcl])
