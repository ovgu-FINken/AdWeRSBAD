import psycopg

from adwersbad.log import setup_logger

logger = setup_logger(__name__)
# TODO: Compare to current db structure and update


def create_weather_table():
    """Create a table in the PostgreSQL database"""
    command = """
            CREATE TABLE IF NOT EXISTS weather (
            time timestamp NOT NULL,
            weather_uid int primary key GENERATED ALWAYS AS IDENTITY,
            weather TEXT NOT NULL,
            location TEXT NOT NULL,
            tod TEXT NOT NULL,
            split TEXT NOT NULL,
            dataset TEXT NOT NULL,
            UNIQUE (time,location)
            )
            """
    return command, None


def create_lidar_table():
    """Create a table in the PostgreSQL database"""
    command = """
        CREATE TABLE IF NOT EXISTS lidar (
            lidar_uid int primary key GENERATED ALWAYS AS IDENTITY,
            weather_uid int NOT NULL,
            points bytea NOT NULL,
            points_downsampled bytea,
            features bytea,
            features_downsampled bytea,
            lidar_id text,
            lidar_parameters text,
            lidar_vehicle_pose text,
            UNIQUE(weather_uid, lidar_id)
        )
        """
    """Add foreign keys to the PostgreSQL database"""
    command_fk = """
        ALTER TABLE lidar
        ADD CONSTRAINT fk_weather
            FOREIGN KEY(weather_uid) REFERENCES weather(weather_uid)
            ON UPDATE CASCADE
            ON DELETE CASCADE
        """
    return command, command_fk


def create_lidar_semantic_table():
    """Create a table in the PostgreSQL database"""
    command = """
        CREATE TABLE IF NOT EXISTS lidar_segmentation (
            lidar_uid int primary key,
            lidar_segmentation bytea,
            lidar_segmentation_downsampled bytea
        )
        """
    command_fk = """
            ALTER TABLE lidar_segmentation 
            ADD CONSTRAINT fk_lidar
            FOREIGN KEY(lidar_uid) REFERENCES lidar(lidar_uid)
            ON UPDATE CASCADE
            ON DELETE CASCADE
            """
    return command, command_fk


def create_lidar_box_table():
    """Create a table in the PostgreSQL database"""
    command = """
        CREATE TABLE IF NOT EXISTS lidar_box (
            lidar_uid int primary key,
            lidar_box text
        )
        """
    command_fk = """
            ALTER TABLE lidar_box 
            ADD CONSTRAINT fk_lidar
            FOREIGN KEY(lidar_uid) REFERENCES lidar(lidar_uid)
            ON UPDATE CASCADE
            ON DELETE CASCADE
            """
    return command, command_fk


def create_camera_table():
    """Create a table in the PostgreSQL database"""
    command = """
        CREATE TABLE IF NOT EXISTS camera (
            camera_uid int primary key GENERATED ALWAYS AS IDENTITY,
            weather_uid int NOT NULL,
            image bytea NOT NULL,
            semantic_labels bytea,
            instance_labels bytea,
            panoptic_labels bytea,
            camera_id text,
            camera_parameters text,
            camera_vehicle_pose text,
            UNIQUE(weather_uid, camera_id)
        )
        """
    """Add foreign keys to the PostgreSQL database"""
    command_fk = """
        ALTER TABLE camera
        ADD CONSTRAINT fk_weather
            FOREIGN KEY (weather_uid) REFERENCES weather(weather_uid)
            ON UPDATE CASCADE
            ON DELETE CASCADE
        """
    return command, command_fk


def create_camera_semantic_table():
    """Create a table in the PostgreSQL database"""
    command = """
        CREATE TABLE IF NOT EXISTS camera_segmentation (
            camera_uid int primary key,
            camera_segmentation bytea
        )
        """
    command_fk = """
            ALTER TABLE camera_segmentation 
            ADD CONSTRAINT fk_camera
            FOREIGN KEY(camera_uid) REFERENCES camera(camera_uid)
            ON UPDATE CASCADE
            ON DELETE CASCADE
            """
    return command, command_fk


def create_results_camera_semantic_table():
    command = """
        CREATE TABLE IF NOT EXISTS results_camera_segmentation (
            camera_uid int primary key,
            prediction bytea,
            iou float4,
            ece float4,
            model text
        )
        """
    command_fk = """
            ALTER TABLE results_camera_segmentation 
            ADD CONSTRAINT fk_camera
            FOREIGN KEY(camera_uid) REFERENCES camera(camera_uid)
            ON UPDATE CASCADE
            ON DELETE CASCADE
            """
    return command, command_fk


def create_results_lidar_semantic_table():
    command = """
        CREATE TABLE IF NOT EXISTS results_lidar_segmentation (
            lidar_uid int primary key,
            prediction bytea,
            prediction_downsampled bytea,
            iou float4,
            iou downsampled float4,
            ece float4,
            ece downsampled float4,
            model text
        )
        """
    command_fk = """
            ALTER TABLE results_lidar_segmentation 
            ADD CONSTRAINT fk_lidar
            FOREIGN KEY(lidar_uid) REFERENCES camera(lidar_uid)
            ON UPDATE CASCADE
            ON DELETE CASCADE
            """
    return command, command_fk


def create_camera_box_table():
    """Create a table in the PostgreSQL database"""
    command = """
        CREATE TABLE IF NOT EXISTS camera_box (
            camera_uid int primary key,
            camera_box text
        )
        """
    command_fk = """
            ALTER TABLE camera_box 
            ADD CONSTRAINT fk_camera
            FOREIGN KEY(camera_uid) REFERENCES camera(camera_uid)
            ON UPDATE CASCADE
            ON DELETE CASCADE
            """
    return command, command_fk


def convert_to_hypertable(table):
    """Convert a table to a hypertable"""
    if table == "weather":
        chunk_time_interval = 3600000
        chunk_column = "time"
    else:
        chunk_time_interval = 1000
        chunk_column = "id"
    command = """
        SELECT create_hypertable('{table}', '{chunk_column}', chunk_time_interval => {chunk_time_interval})
        """.format(
        table=table, chunk_column=chunk_column, chunk_time_interval=chunk_time_interval
    )
    return command


def drop_tables():
    command = """
        DROP TABLE IF EXISTS
        weather,
        lidar, lidar_box, lidar_segmentation,
        camera, camera_box, camera_segmentation,
        results_camera_segmentation, results_lidar_segmentation
        """
    return command


def execute_commands(cur, conn, commands):
    """Execute the given SQL commands in a transaction"""
    for command, foreign_key in commands:
        try:
            # Execute the table creation command
            logger.info("Executing SQL command: %s", command.strip())
            cur.execute(command)
            if foreign_key:
                # If there's a foreign key constraint, execute it
                logger.info("Executing SQL command: %s", foreign_key.strip())
                cur.execute(foreign_key)
            conn.commit()
        except psycopg.errors.DuplicateTable:
            conn.rollback()
        except psycopg.errors.DuplicateObject:
            conn.rollback()


if __name__ == "__main__":
    from adwersbad.config import config

    logger = setup_logger(module_name="createtable")
    # Database connection setup
    params = config(section="psycopg@docker")
    conn = psycopg.connect(**params)
    cur = conn.cursor()

    # DROP EXISITNG
    cur.execute(drop_tables())
    conn.commit()

    # List of all table creation functions
    tables = [
        create_weather_table(),
        create_lidar_table(),
        create_lidar_semantic_table(),
        create_lidar_box_table(),
        create_camera_table(),
        create_camera_semantic_table(),
        create_results_camera_semantic_table(),
        create_results_lidar_semantic_table(),
        create_camera_box_table(),
    ]

    # Execute the table creation commands
    execute_commands(cur, conn, tables)

    # Close connection
    conn.close()
