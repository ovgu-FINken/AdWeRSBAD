from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from io import BytesIO
from typing import List

import networkx as nx
import psycopg
from psycopg import sql
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from adwersbad.config import config

# from itertools import pairwise
from adwersbad.log import setup_logger

_connectionPool = None
_connection = None

logger = setup_logger(__name__)


# logging.basicConfig(
#     level=logging.INFO,  # Change to DEBUG for more detailed output
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("db.log"),  # Log to a file
#         # logging.StreamHandler()  # Also log to the console
#     ]
# )
# logger = logging.getLogger(__name__)
def pairwise(iterable):
    # pairwise('ABCDEFG') → AB BC CD DE EF FG
    iterator = iter(iterable)
    a = next(iterator, None)
    for b in iterator:
        yield a, b
        a = b


@dataclass
class Table:
    name: str
    columns: List[str] = field(default_factory=list)
    child_tables: dict[Table, List[str]] = field(default_factory=dict)
    parent_tables: dict[Table, List[str]] = field(default_factory=dict)


def get_table_by_name(list_of_tables: List[Table], name: str) -> Table:
    for t in list_of_tables:
        if t.name == name:
            return t
    raise ValueError(f"No table with name {name}, available tables: {tables}")


def setup_tables(conn):
    infocursor = conn.cursor()
    infocursor.execute(
        """SELECT table_name FROM information_schema.tables
           WHERE table_schema = 'public'"""
    )
    tables = []
    available_columns = []
    column_to_table_map = defaultdict(list)
    for t in infocursor.fetchall():
        table_name = t[0]
        table = Table(table_name)
        tables.append(table)
        tempcursor = conn.cursor()
        tempquery = sql.SQL(
            """SELECT attname
                    FROM   pg_attribute
                    WHERE  attrelid = {}::regclass  -- table name optionally schema-qualified
                    AND    attnum > 0
                    AND    NOT attisdropped
                    ORDER  BY attnum;
            """
        ).format(table_name)
        for x in tempcursor.execute(tempquery):
            table.columns.append(x[0])
            column_to_table_map[x[0]].append(table.name)
            available_columns.append(x[0])
    infocursor.close()
    return tables, column_to_table_map, available_columns


def get_table_graph(conn, tables):
    G = nx.DiGraph()
    # thanks SO
    constraintcursor = conn.cursor()
    for table in tables:
        query = sql.SQL(
            """
                    select 
                        att2.attname as "child_column", 
                        cl.relname as "parent_table", 
                        att.attname as "parent_column"
                    from
                       (select 
                            unnest(con1.conkey) as "parent", 
                            unnest(con1.confkey) as "child", 
                            con1.confrelid, 
                            con1.conrelid,
                            con1.conname
                        from 
                            pg_class cl
                            join pg_namespace ns on cl.relnamespace = ns.oid
                            join pg_constraint con1 on con1.conrelid = cl.oid
                        where
                            cl.relname = {}
                            and ns.nspname = 'public'
                            and con1.contype = 'f'
                       ) con
                       join pg_attribute att on
                           att.attrelid = con.confrelid and att.attnum = con.child
                       join pg_class cl on
                           cl.oid = con.confrelid
                       join pg_attribute att2 on
                           att2.attrelid = con.conrelid and att2.attnum = con.parent"""
        ).format(table.name)
        for res in constraintcursor.execute(query).fetchall():
            col, ref_table, ref_col = res
            parent = get_table_by_name(tables, ref_table)
            if G.has_edge(table.name, ref_table):
                G.get_edge_data(table.name, ref_table)["columns"].append((col, ref_col))
                table.parent_tables[parent.name].append([ref_col])
                G.get_edge_data(ref_table, table.name)["columns"].append((ref_col, col))
                parent.child_tables[table.name].append([col])
            else:
                G.add_edge(table.name, ref_table, columns=[(col, ref_col)])
                G.add_edge(ref_table, table.name, columns=[(ref_col, col)])
                table.parent_tables[parent.name] = [ref_col]
                parent.child_tables[table.name] = [col]
            # try:
            #    table.parent_tables[parent].append(ref_col)
            # except KeyError:
            #    raise
    return G


# pos=nx.spring_layout(G)
# nx.draw(G, pos=pos, with_labels=True)
# nx.draw_networkx_edge_labels(G,pos,edge_labels=nx.get_edge_attributes(G,'columns'))
# plt.show()


def get_path_from_to_table(G: nx.DiGraph, t1: Table, t2: Table):
    logger.debug(f"Building join path from {t1} to {t2}")
    p = nx.shortest_path(G, source=t1, target=t2)
    joins = []
    for x, y in pairwise(p):
        logger.debug(f"{x} to {y}:")
        join_columns = G.get_edge_data(x, y)["columns"]
        logger.debug(join_columns)
        joins.append((x, y, join_columns))
    return joins


def save_weather_to_db(
    conn: psycopg.Connection,
    time: datetime,
    location: str,
    weather: str,
    tod: str,
    dataset: str,
    split: str,
) -> int:
    """
    Saves weather data to the database.

    Args:
        conn (psycopg.Connection): Database connection object.
        time (int): Timestamp of the weather data.
        location (str): Location of the weather data.
        weather (str): Weather condition description.
        tod (str): Time of day.
        split (str): Dataset split (e.g., training, validation).

    Returns:
        int: Unique identifier for the weather record in the database. Returns -1
        if an error occurs.
    """
    cur = conn.cursor()
    insert_weather = """WITH ins as (
                               INSERT INTO weather (
                               time, location, weather, tod, dataset, split)
                               VALUES (%s, %s, %s, %s, %s, %s) 
                               ON CONFLICT DO NOTHING
                               RETURNING weather_uid
                               )
                        SELECT weather_uid from ins
                        UNION ALL
                        SELECT weather_uid from weather where time = %s and location = %s;
                     """
    cur.execute(
        insert_weather, (time, location, weather, tod, dataset, split, time, location)
    )
    res = cur.fetchone()
    if res:
        weather_uid = res[0]
    else:
        # TODO different handling if the cause was duplicate data (conflict)
        # in that case we cant to check if the image and pcl data is present in the table before aborting the following insertion steps
        logger.error(
            f"Database error while saving weather data with \
                     {time=}, {location=}, {weather=}, {tod=}, {dataset=}, {split=}"
        )
        weather_uid = -1
    return weather_uid


def save_image_to_db(
    conn: psycopg.Connection,
    weather_uid: int,
    image: BytesIO,
    camera_id: str,
    camera_params: str,
    vehicle_pose: str,
) -> int:
    """
    Saves camera image data to the database.

    Args:
        conn (psycopg.Connection): Database connection object.
        weather_uid (int): Unique identifier for the weather record.
        image (BytesIO): Image data in bytes.
        camera_id (str): Camera identifier.
        camera_params (str): JSON string of the camera matrices

    Returns:
        int: Unique identifier for the camera record in the database.
             Returns -1 if an error occurs.
    """
    cur = conn.cursor()
    insert_camera_query = """WITH ins AS (
                                INSERT INTO camera (
                                    weather_uid, 
                                    image, camera_id,
                                    camera_parameters,
                                    camera_vehicle_pose)
                                VALUES (
                                    %s, 
                                    %s, %s,
                                    %s,
                                    %s)
                                ON CONFLICT(weather_uid, camera_id)
                                DO UPDATE SET
                                    image = EXCLUDED.image,
                                    camera_parameters = EXCLUDED.camera_parameters,
                                    camera_vehicle_pose = EXCLUDED.camera_vehicle_pose
                                RETURNING camera_uid
                            )
                            SELECT camera_uid FROM ins
                            UNION ALL
                            SELECT camera_uid FROM camera 
                            WHERE weather_uid = %s and camera_id = %s;
                          """
    cur.execute(
        insert_camera_query,
        (
            weather_uid,
            image.getvalue(),
            camera_id,
            camera_params,
            vehicle_pose,
            weather_uid,
            camera_id,
        ),
    )
    res = cur.fetchone()
    if res:
        camera_uid = res[0]
    else:
        logger.error(
            f"Database error while saving camera data with \
                    {weather_uid=}, {time=}, {location=}, {camera_id=}"
        )
        camera_uid = -1
    return camera_uid


def save_image_labels_to_db(
    conn: psycopg.Connection, camera_uid: int, segmentation: BytesIO | None, boxes: str
) -> None:
    """
    Saves camera image labels (segmentation and bounding boxes) to the database.

    Args:
        conn (psycopg.Connection): Database connection object.
        camera_uid (int): Unique identifier for the camera record.
        segmentation (BytesIO): Segmentation data in bytes, None if no segmentation data available
        boxes (str): JSON string of bounding box data.
    """
    cur = conn.cursor()
    if segmentation:
        insert_camseg = """
            INSERT INTO camera_segmentation (
                camera_uid, 
                camera_segmentation
            )
            VALUES (%s, %s)
            ON CONFLICT (camera_uid) DO UPDATE 
            SET camera_segmentation = EXCLUDED.camera_segmentation
            """
        cur.execute(insert_camseg, (camera_uid, segmentation.getvalue()))
    if boxes:
        insert_cambox = """
            INSERT INTO camera_box (
                camera_uid, 
                camera_box
            )
            VALUES (%s, %s)
            ON CONFLICT (camera_uid) DO UPDATE 
            SET camera_box = EXCLUDED.camera_box
        """
        cur.execute(insert_cambox, (camera_uid, boxes))
    return


def save_pointcloud_to_db(
    conn: psycopg.Connection,
    weather_uid: int,
    points: BytesIO,
    features: BytesIO,
    lidar_name: str,
    lidar_params: str,
    vehicle_pose: str,
):
    """
    Saves point cloud data to the database.

    Args:
        conn (psycopg.Connection): Database connection object.
        weather_uid (int): Unique identifier for the weather record.
        points (BytesIO): Point cloud data in bytes.
        features (BytesIO): Feature data in bytes.
        lidar_name (str): Name or id of the lidar sensor
        lidar_params (str): JSON string of lidar calibration parameters.

    Returns:
        int: Unique identifier for the lidar record in the database.
             Returns -1 if an error occurs.
    """
    cur = conn.cursor()
    insert_lidar = """
                WITH ins AS (
                   INSERT INTO lidar (
                   weather_uid, 
                   points, features,
                   lidar_id,
                   lidar_parameters,
                   lidar_vehicle_pose)
                   VALUES (%s, 
                   %s, %s,
                   %s,
                   %s,
                   %s
                   ) 
                   ON CONFLICT (weather_uid, lidar_id) DO UPDATE 
                   SET 
                       points = EXCLUDED.points,
                       features = EXCLUDED.features,
                       lidar_parameters = EXCLUDED.lidar_parameters,
                       lidar_vehicle_pose = EXCLUDED.lidar_vehicle_pose
                   RETURNING lidar_uid
                )
                SELECT lidar_uid FROM ins
                UNION ALL
                SELECT lidar_uid from lidar 
                WHERE weather_uid = %s and lidar_id = %s
               """
    cur.execute(
        insert_lidar,
        (
            weather_uid,
            points.getvalue(),
            features.getvalue(),
            lidar_name,
            lidar_params,
            vehicle_pose,
            weather_uid,
            lidar_name,
        ),
    )
    res = cur.fetchone()
    if res:
        lidar_uid = res[0]
    else:
        logger.error(
            f"Database error while saving lidar data with \
                    {weather_uid=}"
        )
        lidar_uid = -1
    return lidar_uid


def save_pointcloud_labels_to_db(
    conn: psycopg.Connection, lidar_uid: int, segmentation: BytesIO, boxes: str
):
    """
    Saves point cloud labels (segmentation and bounding boxes) to the database.

    Args:
        conn (psycopg.Connection): Database connection object.
        lidar_uid (int): Unique identifier for the lidar record.
        segmentation (BytesIO): Segmentation data in bytes.
        boxes (str): JSON string of bounding box data.
    """
    cur = conn.cursor()
    if segmentation:
        insert_lidarseg = """
                    INSERT INTO lidar_segmentation (
                        lidar_uid, 
                        lidar_segmentation
                    )
                    VALUES (%s, %s)
                    ON CONFLICT (lidar_uid) DO UPDATE
                    SET 
                    lidar_uid = EXCLUDED.lidar_uid,
                    lidar_segmentation=EXCLUDED.lidar_segmentation
                    """
        cur.execute(insert_lidarseg, (lidar_uid, segmentation.getvalue()))
    if boxes:
        insert_cambox = """
                       INSERT INTO lidar_box (
                       lidar_uid, 
                       lidar_box)
                       VALUES (%s, 
                       %s)
                       ON CONFLICT (lidar_uid) DO UPDATE
                       SET lidar_box = EXCLUDED.lidar_box
                       returning lidar_uid
                       """
        cur.execute(insert_cambox, (lidar_uid, boxes))
    return


def update_split(conn: psycopg.Connection, uid: int, split: str):
    cur = conn.cursor()
    upd = """
        UPDATE weather
        SET split=%s
        WHERE weather_uid=%s
        """
    cur.execute(upd, (split, uid))
    return


def get_connection(dbinfo: str) -> psycopg.Connection:
    try:
        params = config(section=dbinfo)
        conn = psycopg.connect(**params)
    except psycopg.DatabaseError as e:
        logger.error(f"error when creating database connection with {dbinfo=}: {e}")
        raise
    return conn


def get_connection_pool(
    dbinfo: str, min_size: int = 4, max_size: int = 12
) -> ConnectionPool:
    """Create and return a connection pool for the database."""
    try:
        params = config(section=dbinfo)
        conninfo = (
            f"dbname={params['dbname']} user={params['user']} host={params['host']} "
        )

        pool = ConnectionPool(conninfo=conninfo, min_size=min_size, max_size=max_size)
    except psycopg.DatabaseError as e:
        logger.error(
            f"Error when creating database connection pool with {dbinfo=}, {params=}: {e}"
        )
        raise
    return pool


def get_async_connection_pool(
    dbinfo: str, min_size: int = 4, max_size: int = 12
) -> ConnectionPool:
    """Create and return a connection pool for the database."""
    try:
        params = config(section=dbinfo)
        conninfo = (
            f"dbname={params['dbname']} user={params['user']} host={params['host']} "
        )

        pool = AsyncConnectionPool(
            conninfo=conninfo, min_size=min_size, max_size=max_size
        )
    except psycopg.DatabaseError as e:
        logger.error(
            f"Error when creating database connection pool with {dbinfo=}, {params=}: {e}"
        )
        raise
    return pool


if __name__ == "__main__":
    from tqdm.contrib.concurrent import process_map

    pool = get_connection_pool("psycopg3@local", max_size=5)
    with pool.connection() as conn:
        tables = setup_tables(conn)
        print(tables)
        table_graph = get_table_graph(conn)
        print(
            get_path_from_to_table(
                table_graph,
                get_table_by_name("results_lidar"),
                get_table_by_name("weather"),
            )
        )
