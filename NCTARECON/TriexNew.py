import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
import paramiko
from tqdm import tqdm
import zipfile
import logging
import numpy as np
from sqlalchemy import create_engine, MetaData, Table, select, insert
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, DateTime
import time
import signal
import atexit


# Load environment variables from .env file
load_dotenv()

# Get environment variables
DATABASE_URI = os.getenv('DATABASE_URI')
SFTP_HOST = os.getenv('SFTP_HOST')
SFTP_USERNAME = os.getenv('SFTP_USERNAME')
SFTP_PASSWORD = os.getenv('SFTP_PASSWORD')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Global variable to store the last processed row details
last_processed_row = {'num_records_processed': 0, 'index_id': 0}
processing_complete = False  # Global flag to track processing completion

def signal_handler(file_name, engine, sig, frame):
    """Signal handler for SIGINT."""
    logging.info(f"KeyboardInterrupt (ID: {sig}) has been caught. Cleaning up...")
    if last_processed_row['num_records_processed'] > 0 and not processing_complete:
        update_checkpoint(engine, file_name, last_processed_row['num_records_processed'], last_processed_row['index_id'], 'interrupted')
    logging.info("Cleanup complete. Exiting...")
    exit(0)

def atexit_handler():
    """Handler to update the checkpoint when the program exits."""
    if last_processed_row['num_records_processed'] > 0 and not processing_complete:
        update_checkpoint(engine, file_name, last_processed_row['num_records_processed'], last_processed_row['index_id'], 'interrupted')
    logging.info("Program exited. Checkpoint updated.")

atexit.register(atexit_handler)

def setup_signal_handlers(file_name, engine):
    """Setup signal handlers for various interruptions."""
    def handler(sig, frame):
        logging.info(f"Signal {sig} received. Cleaning up...")
        if last_processed_row['num_records_processed'] > 0 and not processing_complete:
            update_checkpoint(engine, file_name, last_processed_row['num_records_processed'], last_processed_row['index_id'], 'interrupted')
        logging.info("Cleanup complete. Exiting...")
        exit(0)

    signals = [signal.SIGINT, signal.SIGTERM]
    for sig in signals:
        signal.signal(sig, handler)

def get_next_file_to_process(engine, sftp, sftp_path):
    """
    Modified approach:
    1) Collect all rows where status='unprocessed' and filename like 'TRX%'.
    2) Parse the date from each filename (e.g. TRXYYYYMMDD.zip).
    3) Sort them by the date in ascending order, pick the oldest.
    4) If no unprocessed is found, then pick the most recent processed date 
       from the checkpoint, increment by one day, and see if the file is on SFTP.
    5) If no records at all, start from default date '20240729'.
    """
    with engine.connect() as conn:
        metadata = MetaData()
        checkpoint_table = Table('tbCheckpoint', metadata, autoload_with=engine)

        # --- 1) Fetch *all* unprocessed TRX files ---
        all_unprocessed = conn.execute(
            select(checkpoint_table)
            .where(checkpoint_table.c.status == 'unprocessed')
            .where(checkpoint_table.c.filename.like('TRX%'))
        ).fetchall()

        # --- 2) If we have any unprocessed, parse their dates from filename ---
        if all_unprocessed:
            # Each row => parse the date from the filename, store (parsed_date, row)
            parsed_unprocessed = []
            for row in all_unprocessed:
                try:
                    # Filename example: TRX20250117.zip
                    # Extract '20250117' from between 'TRX' and '.zip'
                    base_name = row.filename.replace('.zip', '')
                    # If your filenames are always TRX + 8-digit date, e.g. TRX20250101
                    date_str = base_name[-8:]  # last 8 chars
                    file_date = datetime.strptime(date_str, "%Y%m%d").date()
                    parsed_unprocessed.append((file_date, row))
                except ValueError:
                    # If filename doesn't match the expected format
                    continue

            # Sort by file_date ascending
            parsed_unprocessed.sort(key=lambda x: x[0])

            # Pick the oldest date
            oldest_date, oldest_row = parsed_unprocessed[0]
            logging.info(f"Found unprocessed files, picking the oldest date: {oldest_row.filename}")
            return oldest_row.filename

        # --- 3) If we get here, there are no unprocessed TRX files ---
        #     So now we look for the most recent processed record.
        last_processed_record = conn.execute(
            select(checkpoint_table)
            .where(checkpoint_table.c.status == 'processed')
            .where(checkpoint_table.c.filename.like('TRX%'))
            .order_by(checkpoint_table.c.CreatedDateTime.desc())
        ).fetchone()

        if last_processed_record:
            last_processed_filename = last_processed_record.filename
            logging.info(f"No unprocessed files. Most recent processed file: {last_processed_filename}")
            try:
                last_processed_date = datetime.strptime(
                    last_processed_filename.split('.')[0][-8:], 
                    "%Y%m%d"
                )
            except ValueError:
                logging.error(f"Could not parse date from last processed filename: {last_processed_filename}")
                return None

            next_file_date = last_processed_date + timedelta(days=1)
            next_file_name = f'TRX{next_file_date.strftime("%Y%m%d")}.zip'
        else:
            # If no records exist at all, start from a default date
            next_file_date = datetime.strptime('20240729', "%Y%m%d")
            next_file_name = f'TRX{next_file_date.strftime("%Y%m%d")}.zip'
            logging.info(f"No processed/unprocessed files found. Starting from default date: {next_file_date}")

        # --- 4) Check if next file is on SFTP ---
        logging.info("Listing files on SFTP server...")
        sftp_files = list_files_on_sftp(sftp, sftp_path)
        logging.info(f"Files on SFTP server: {sftp_files}")

        if next_file_name in sftp_files:
            # Create a new checkpoint entry for the next file if it exists on SFTP
            conn.execute(
                insert(checkpoint_table).values(
                    filename=next_file_name,
                    numrecordsprocessed=0,
                    indexid=0,
                    status='unprocessed',
                    CreatedDateTime=datetime.now(),
                    UpdatedDateTime=datetime.now()
                )
            )
            logging.info(f"Next file to process (post-processed scenario): {next_file_name}")
            return next_file_name
        else:
            logging.error(f"Next file {next_file_name} not found on SFTP server.")
            return None


def list_files_on_sftp(sftp, sftp_path):
    """List files in the SFTP server directory."""
    try:
        files = sftp.listdir(sftp_path)
        return files
    except Exception as e:
        logging.error(f"An error occurred while listing files on SFTP server: {e}")
        return []

def connect_to_sftp():
    """Connect to the SFTP server and return the SFTP client."""
    try:
        transport = paramiko.Transport((SFTP_HOST, 22))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        return sftp, transport
    except Exception as e:
        logging.error(f"An error occurred while connecting to SFTP server: {e}")
        return None, None

def update_transaction_status(df):
    """
    Update the TransactionStatus to 'Batched' where TransactionDeliveryDateTime
    is not '1900-01-01 00:00:00'.
    """
    # Define the condition
    condition = df['TransactionDeliveryDateTime'] != '1900-01-01 00:00:00'

    # Apply the update
    df.loc[condition, 'TransactionStatus'] = 'Batched'

    return df


def update_mir_reject_status(df):
    """
    Update the TransactionStatus to 'MIR Reject' where MIRReject is greater than or equal to 1.
    Additionally, ensure all NaN values remain as NaN and any 0 or 0.0 values are converted to NaN.
    """
    # Convert 'MIRReject' column to float, ensuring NaNs are handled
    df['MIRReject'] = pd.to_numeric(df['MIRReject'], errors='coerce')
    
    # Replace 0 and 0.0 with NaN (to later be treated as NULL)
    df['MIRReject'] = df['MIRReject'].replace(0, np.nan)
    
    # Define the condition for 'MIR Reject' status
    condition = df['MIRReject'] >= 1

    # Apply the update to TransactionStatus
    df.loc[condition, 'TransactionStatus'] = 'MIR Reject'

    return df


def update_unknown_status(df):
    """
    Update the TransactionStatus to 'Unknown' where:
    1. TransactionType is 'T' or 'V'
    2. TransactionDeliveryDateTime is '1900-01-01 00:00:00.000000' or NULL
    3. TransactionStatus is 'Batched'
    """

    # Ensure the datetime format in the DataFrame is compatible with the SQL precision level
    df['TransactionDeliveryDateTime'] = pd.to_datetime(df['TransactionDeliveryDateTime'], errors='coerce')

    # Define the condition for TransactionType 'T'
    condition_T = (
        (df['TransactionType'] == 'T') &
        ((df['TransactionDeliveryDateTime'] == pd.Timestamp('1900-01-01 00:00:00.000000')) | df['TransactionDeliveryDateTime'].isnull()) &
        (df['TransactionStatus'] == 'Batched')
    )

    # Define the condition for TransactionType 'V'
    condition_V = (
        (df['TransactionType'] == 'V') &
        ((df['TransactionDeliveryDateTime'] == pd.Timestamp('1900-01-01 00:00:00.000000')) | df['TransactionDeliveryDateTime'].isnull()) &
        (df['TransactionStatus'] == 'Batched')
    )

    # Apply the update for both conditions
    df.loc[condition_T | condition_V, 'TransactionStatus'] = 'Unknown'

    return df





def main():
    global engine, file_name, processing_complete
    try:
        # Create database engine
        engine = create_engine(DATABASE_URI)

        while True:
            # Connect to SFTP server
            sftp, transport = connect_to_sftp()
            if sftp is None or transport is None:
                logging.error("Failed to establish SFTP connection. Exiting.")
                break

            logging.info("Determining the next file to process...")
            # Determine the next file to process
            file_name = get_next_file_to_process(engine, sftp, '/Dataingest/')

            if not file_name:
                logging.info("No more files to process.")
                sftp.close()
                transport.close()
                break

            logging.info(f"Processing file: {file_name}")

            # Download CSV file from within ZIP file on SFTP server
            sftp_full_path = f'/Dataingest/{file_name}'
            local_path = r'C:\Users\KhokharA\Documents\LoadData\RoadsideCSVs'  # Your local path

            logging.info(f"Downloading CSV from within ZIP file from SFTP server: {sftp_full_path}")
            local_csv_path = download_csv_from_zip_on_sftp(sftp, sftp_full_path, local_path)

            if not local_csv_path:
                logging.error("Failed to download CSV file from ZIP on SFTP.")
                sftp.close()
                transport.close()
                break

            # After loading data
            triex_recon_df = load_csv_data(local_csv_path)

            # Make sure to clean 'None' strings that might still be present
            triex_recon_df.replace({'None': np.nan, '': np.nan}, inplace=True)

            if triex_recon_df.empty:
                logging.error("CSV file could not be loaded or is empty.")
                sftp.close()
                transport.close()
                break

            # Continue with the rest of your processing steps...

            # Rename columns and get column mapping
            metadata = MetaData()
            metadata.reflect(bind=engine)
            with engine.connect() as conn:
                # Check existing checkpoint for the file
                checkpoint_table = Table('tbCheckpoint', metadata, autoload_with=engine)
                checkpoint_record = conn.execute(
                    select(checkpoint_table).where(checkpoint_table.c.filename == file_name)
                ).fetchone()

                if checkpoint_record:
                    logging.info(f"Checkpoint exists: {checkpoint_record}")
                    last_processed_index = checkpoint_record.numrecordsprocessed  # Start from the last processed row

                    if last_processed_index >= len(triex_recon_df):
                        logging.info("All rows in the file have been processed. No new rows to process.")
                        sftp.close()
                        transport.close()
                        continue
                else:
                    # logging.info("No checkpoint found. Processing from the beginning.")
                    last_processed_index = 0

                triex_recon_df = triex_recon_df.iloc[last_processed_index:]

                if triex_recon_df.empty:
                    logging.info("No new rows to process.")
                    sftp.close()
                    transport.close()
                    continue

                # Rename columns based on mapping table
                rename_columns(conn, triex_recon_df)

                # Rename history columns
                #history_column_df = fetch_history_table(conn, metadata)
                #rename_history_columns(conn, history_column_df)

                # Fetch step one mapping data from SQL Server
                step_one_mapping_df = fetch_step_one_mapping(conn)

                # Perform step one mapping
                triex_recon_df = perform_step_one_mapping(triex_recon_df, step_one_mapping_df)
                print("Step one mapping complete")
                print("Columns in df after Step 1:", triex_recon_df.columns.tolist())


                # Fetch step two mapping data from SQL Server
                step_two_mapping_df = fetch_step_two_mapping(conn)

                # Perform step two mapping
                triex_recon_df = perform_step_two_mapping(triex_recon_df, step_two_mapping_df)
                print("Step two mapping complete")

                # Fetch Step 3 Mapping data from SQL Server
                step_three_mapping_df = fetch_step_three_mapping(conn)

                # Perform Step 3 mapping
                triex_recon_df = perform_step_three_mapping(triex_recon_df, step_three_mapping_df)
                print("Step three mapping complete")

                # Data Transformation and Status Updates
                triex_recon_df = update_transaction_status(triex_recon_df)
                triex_recon_df = update_mir_reject_status(triex_recon_df)
                triex_recon_df = update_unknown_status(triex_recon_df)  # New function added here

                # Cast all values to strings
                triex_recon_df = triex_recon_df.astype(str)
                triex_recon_df['TransactionID'] = triex_recon_df['TransactionID'].astype(np.int64)

                # Replace 'nan' strings with None
                triex_recon_df.replace('nan', None, inplace=True)

                # Setup signal handlers
                setup_signal_handlers(file_name, engine)

                # Insert transformed data into SQL Server
                insert_data_into_sql(engine, triex_recon_df, file_name)

                processing_complete = True  # Mark processing as complete if no exceptions occur

            # Close SFTP connection
            sftp.close()
            transport.close()

    except Exception as e:
        logging.error(f"An error occurred: {e}")



def update_checkpoint(engine, file_name, num_records_processed, index_id, status):
    """Update the tbCheckpoint table with the current processing status."""
    try:
        Session = sessionmaker(bind=engine)
        session = Session()
        metadata = MetaData()
        metadata.reflect(bind=engine)
        checkpoint_table = Table('tbCheckpoint', metadata, autoload_with=engine)
        
        # Get the current timestamp
        current_timestamp = datetime.now()

        # Convert numpy.int64 to Python int
        num_records_processed = int(num_records_processed)
        index_id = int(index_id)
        
        # Check if there is an existing record for the file
        existing_record = session.execute(select(checkpoint_table).where(checkpoint_table.c.filename == file_name)).fetchone()
        
        if existing_record:
            # Update the existing record
            update_query = (
                checkpoint_table.update()
                .where(checkpoint_table.c.filename == file_name)
                .values(
                    numrecordsprocessed=num_records_processed,
                    indexid=index_id,
                    status=status,
                    UpdatedDateTime=current_timestamp
                )
            )
            session.execute(update_query)
        else:
            # Insert a new record
            insert_query = checkpoint_table.insert().values(
                filename=file_name,
                numrecordsprocessed=num_records_processed,
                indexid=index_id,
                status=status,
                CreatedDateTime=current_timestamp,
                UpdatedDateTime=current_timestamp
            )
            session.execute(insert_query)
        
        session.commit()
        logging.info(f"Checkpoint updated: {file_name, num_records_processed, index_id, status}")
    except Exception as e:
        logging.error(f"Failed to update checkpoint: {e}")
    finally:
        session.close()

def insert_data_into_sql(engine, df, file_name, batch_size=1000, max_retries=5, retry_wait=1):
    """Insert or update transformed data into SQL Server using bulk inserts and log updates to a history table."""
    global last_processed_row
    try:
        # Define the table names
        table_name = 'tbtriexrecondailydetail'
        history_table_name = 'tbtriexrecondailydetailhist'
        
        # Add UpdatedBy and UpdatedTimeStamp columns to the DataFrame
        current_user = 'NCTA_Reports'
        current_timestamp = datetime.now()
        df['UpdatedBy'] = current_user
        df['UpdatedTimeStamp'] = current_timestamp

        # Ensure datetime columns are properly converted to datetime objects and then formatted
        datetime_columns = ['TransactionDateTime', 'DispositionReceiptDateTime', 'TransactionDeliveryDateTime', 'TimeRecievedByRSS']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')  # Convert to datetime, coerce errors to NaT
                df[col] = df[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if not pd.isnull(x) else None)

        # Replace 'None' strings and NaN with actual None to reflect NULL in SQL
        df.replace([np.nan, 'None', 'nan'], None, inplace=True)

        # Create a SQLAlchemy session
        Session = sessionmaker(bind=engine)
        session = Session()
        metadata = MetaData()
        metadata.reflect(bind=engine)
        table = Table(table_name, metadata, autoload_with=engine)
        history_table = Table(history_table_name, metadata, autoload_with=engine)

        # Process data row by row with a progress bar
        with tqdm(total=df.shape[0], desc="Inserting Data", unit="row") as pbar:
            insert_batch_with_duplicate_handling(session, table, history_table, df, file_name, engine, max_retries, retry_wait, pbar)

        if not df.empty:
            update_checkpoint(engine, file_name, len(df), df['TransactionID'].iloc[-1], 'processed')
        logging.info("Data inserted/updated successfully.")

    except IntegrityError as ie:
        logging.error(f"Integrity error: {ie}")
    except Exception as e:
        logging.error(f"Error inserting data: {e}")
    finally:
        session.close()
        logging.info("Database session closed.")




def insert_batch_with_duplicate_handling(
    session,
    table,
    history_table,
    batch,
    file_name,
    engine,
    max_retries=5,
    retry_wait=1,
    pbar=None
):
    """
    Insert or update data in 'table' with duplicate (TransactionID) handling,
    processing rows in batches of 100 for faster insert performance.
    Uses nested transactions (savepoints) so a single row failure
    doesn't blow up the entire chunk transaction.
    """

    import pandas as pd
    from sqlalchemy import select
    from sqlalchemy.exc import IntegrityError, OperationalError
    import time
    import logging

    global last_processed_row

    records = batch.to_dict("records")  # Each row -> dict
    rows_processed_since_last_checkpoint = 0

    CHUNK_SIZE = 100  # how many rows per chunk

    for chunk_start in range(0, len(records), CHUNK_SIZE):
        chunk = records[chunk_start : chunk_start + CHUNK_SIZE]

        # Begin a single transaction for this entire chunk
        session.begin()  # main transaction for up to 100 rows

        for i, row_dict in enumerate(chunk):
            idx = chunk_start + i  # absolute row index

            # We can do multiple attempts for deadlock handling
            retries = 0
            while retries < max_retries:
                try:
                    # Begin a nested transaction (savepoint) for this single row
                    with session.begin_nested():
                        # Attempt to insert
                        session.execute(table.insert().values(row_dict))

                    # If insertion succeeded, break out of retry loop
                    break

                except IntegrityError:
                    # This means a duplicate or constraint violation
                    session.rollback()  # rollback just this sub-transaction

                    # Handle duplicates
                    existing_row = session.execute(
                        select(table)
                        .with_hint(table, "WITH (NOLOCK)", dialect_name="mssql")
                        .where(table.c.TransactionID == row_dict["TransactionID"])
                    ).fetchone()

                    if existing_row:
                        existing_row_dict = dict(zip(table.columns.keys(), existing_row))

                        # Example: parse relevant date fields
                        existing_disposition_date = pd.to_datetime(
                            existing_row_dict.get("DispositionReceiptDateTime"), errors="coerce"
                        )
                        new_disposition_date = pd.to_datetime(
                            row_dict.get("DispositionReceiptDateTime"), errors="coerce"
                        )

                        existing_transaction_date = pd.to_datetime(
                            existing_row_dict.get("TransactionDeliveryDateTime"), errors="coerce"
                        )
                        new_transaction_date = pd.to_datetime(
                            row_dict.get("TransactionDeliveryDateTime"), errors="coerce"
                        )

                        new_status = row_dict.get("TransactionStatus")

                        # Begin another sub-transaction for handling duplicates
                        with session.begin_nested():

                            # 1) If the new record is in certain statuses => overwrite existing
                            if new_status in ["Batched", "MIR Reject", "Duplicate"]:
                                session.execute(history_table.insert().values(existing_row_dict))
                                session.execute(
                                    table.update()
                                    .where(table.c.TransactionID == row_dict["TransactionID"])
                                    .values(row_dict)
                                )

                            # 2) If both existing and new are missing those date fields
                            elif (
                                pd.isnull(existing_disposition_date)
                                and pd.isnull(existing_transaction_date)
                                and pd.isnull(new_disposition_date)
                                and pd.isnull(new_transaction_date)
                            ):
                                session.execute(history_table.insert().values(existing_row_dict))
                                session.execute(
                                    table.update()
                                    .where(table.c.TransactionID == row_dict["TransactionID"])
                                    .values(row_dict)
                                )

                            # 3) If new dates are strictly newer => overwrite existing
                            elif existing_disposition_date is None or (
                                new_disposition_date is not None
                                and new_disposition_date > existing_disposition_date
                            ):
                                session.execute(history_table.insert().values(existing_row_dict))
                                session.execute(
                                    table.update()
                                    .where(table.c.TransactionID == row_dict["TransactionID"])
                                    .values(row_dict)
                                )
                            elif existing_transaction_date is None or (
                                new_transaction_date is not None
                                and new_transaction_date > existing_transaction_date
                            ):
                                session.execute(history_table.insert().values(existing_row_dict))
                                session.execute(
                                    table.update()
                                    .where(table.c.TransactionID == row_dict["TransactionID"])
                                    .values(row_dict)
                                )

                            # 4) NEW LOGIC: if *all* date/time checks are exactly the same,
                            #    treat the new row as the "newer" version. 
                            elif (
                                existing_disposition_date == new_disposition_date
                                and existing_transaction_date == new_transaction_date
                                # Add more equality checks here if needed, e.g. same amounts, etc.
                            ):
                                # Move the old row to history, store the new row in the main table
                                session.execute(history_table.insert().values(existing_row_dict))
                                session.execute(
                                    table.update()
                                    .where(table.c.TransactionID == row_dict["TransactionID"])
                                    .values(row_dict)
                                )

                            else:
                                # If the new row is "older" or doesn't meet other criteria,
                                # add it to the history only
                                session.execute(history_table.insert().values(row_dict))

                    # Done handling the duplicate; break out of retry loop
                    break

                except OperationalError as oe:
                    # Handle deadlock victims with retries
                    if "deadlock victim" in str(oe):
                        retries += 1
                        logging.warning(f"Deadlock detected. Retrying {retries}/{max_retries}...")
                        session.rollback()
                        time.sleep(retry_wait * (2 ** retries))
                        continue  # retry
                    else:
                        session.rollback()
                        raise

                except Exception as e:
                    session.rollback()
                    raise

            # Checkpoint logic every 1000 rows
            rows_processed_since_last_checkpoint += 1
            if rows_processed_since_last_checkpoint >= 1000:
                update_checkpoint(engine, file_name, idx + 1, row_dict["TransactionID"], "unprocessed")
                rows_processed_since_last_checkpoint = 0

            # Update global last_processed_row
            last_processed_row["num_records_processed"] = idx + 1
            last_processed_row["index_id"] = row_dict["TransactionID"]

            if pbar is not None:
                pbar.update(1)

        # Commit the chunk (up to 100 rows)
        session.commit()

    # Final checkpoint after all records
    if records:
        final_idx = len(records) - 1
        last_row = records[final_idx]
        update_checkpoint(engine, file_name, final_idx + 1, last_row["TransactionID"], "unprocessed")





def download_csv_from_zip_on_sftp(sftp, sftp_path, local_path):
    """Download CSV file from within ZIP file on SFTP server to local path."""
    try:
        # Open the ZIP file on the SFTP server
        with sftp.open(sftp_path, 'rb') as zip_file:
            with zipfile.ZipFile(zip_file) as zip_ref:
                # Get the list of files in the ZIP
                zip_file_list = zip_ref.namelist()
                # Find the CSV file in the ZIP
                csv_file_name = next((name for name in zip_file_list if name.endswith('.csv')), None)
                if not csv_file_name:
                    logging.error("No CSV file found in the ZIP archive.")
                    return None
                # Extract the CSV file content
                with zip_ref.open(csv_file_name) as csv_file:
                    local_file_path = os.path.join(local_path, os.path.basename(csv_file_name))
                    with open(local_file_path, 'wb') as local_file:
                        local_file.write(csv_file.read())
        return local_file_path

    except Exception as e:
        logging.error(f"An error occurred while extracting CSV file from ZIP archive on SFTP server: {e}")
        return None

def load_csv_data(csv_path):
    """Load CSV data into a DataFrame and clean up 'None' strings."""
    df = pd.read_csv(csv_path, dtype=str)  # Load as strings

    # Convert 'TransactionDeliveryDateTime' to datetime, ensuring correct precision
    #df['TransactionDeliveryDateTime'] = pd.to_datetime(df['TransactionDeliveryDateTime'], errors='coerce')

    # Replace 'None' and empty strings with NaN
    df.replace({'None': np.nan, '': np.nan}, inplace=True)

    return df



def rename_columns(conn, df):
    """Rename columns in DataFrame according to mapping table and in SQL table."""
    column_mapping_df = fetch_column_mapping(conn)
    column_mapping = {}
    for _, row in column_mapping_df.iterrows():
        old_column_name = row['RoadsideColumnName']
        new_column_name = row['TransformedColumnName']
        if old_column_name in df.columns:
            column_mapping[old_column_name] = new_column_name
            df.rename(columns={old_column_name: new_column_name}, inplace=True)
            try:
                alter_query = f"EXEC sp_rename 'tbtriexrecondailydetail.{old_column_name}', '{new_column_name}', 'COLUMN'"
                conn.execute(alter_query)
            except Exception as e:
                logging.warning(f"Failed to rename column in SQL: {e}")
                continue
    return column_mapping

def fetch_column_mapping(conn):
    """Fetch column renaming mapping data from SQL Server."""
    query = "SELECT RoadsideColumnName, TransformedColumnName FROM tbMappingTableColumns"
    return pd.read_sql(query, conn)

def fetch_history_table(conn, metadata):
    """Fetch mapping data from SQL Server."""
    tb_history = Table('tbtriexrecondailydetailhist', metadata, autoload_with=conn)
    query = select(tb_history)
    return pd.read_sql(query, conn)

def rename_history_columns(conn, df):
    """Rename columns in DataFrame according to mapping table and in SQL table."""
    column_mapping_df = fetch_column_mapping(conn)
    for _, row in column_mapping_df.iterrows():
        old_column_name = row['RoadsideColumnName']
        new_column_name = row['TransformedColumnName']
        if old_column_name in df.columns:
            df.rename(columns={old_column_name: new_column_name}, inplace=True)
            try:
                alter_query = f"EXEC sp_rename 'tbtriexrecondailydetailhist.{old_column_name}', '{new_column_name}', 'COLUMN'"
                conn.execute(alter_query)
            except Exception as e:
                logging.warning(f"Failed to rename history column in SQL: {e}")
                continue

def fetch_step_one_mapping(conn):
    """Fetch mapping data from SQL Server."""
    query = "SELECT * FROM NCTARECONPROD.dbo.tbMappingTableStep1"
    return pd.read_sql(query, conn)
    
"""
def perform_step_one_mapping(df, mapping_df):
    columns_to_string = mapping_df['RoadsideColumnA'].dropna().unique().tolist()
    for col in columns_to_string:
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].astype(str)
    
    transformed_columns = mapping_df['TransformedColumn'].dropna().unique().tolist()
    for tcol in transformed_columns:
        if tcol not in df.columns:
            df[tcol] = None

    mapping_df.fillna("NULL", inplace=True)
    mapping_df['RoadsideValueA'] = mapping_df['RoadsideValueA'].astype(str)
    mapping_df['TransformedValue'] = mapping_df['TransformedValue'].replace(["NONE", "NULL"], None)
    
    for col_search, val_search, col_replace, val_replace in mapping_df.values:
        if val_search == "NULL":
            mask = df[col_search].isnull()
        else:
            mask = (df[col_search] == val_search)
        df.loc[mask, col_replace] = val_replace
    
    df.replace('None', None, inplace=True)
    return df
    """

from tqdm import tqdm
import pandas as pd
import numpy as np

def perform_step_one_mapping(df, mapping_df):
    """
    Vectorized approach that only applies mappings where Roadway == 'T33'.
    We also only transform rows in df where df['Roadway'] == 'T33'.
    """

    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    # 1) Filter the mapping table to Roadway == 'T33'
    if 'Roadway' in mapping_df.columns:
        mapping_df = mapping_df[mapping_df['Roadway'] == 'T33']
    else:
        # If there's no Roadway column in mapping_df, no mapping is done
        return df

    # If no rows remain, nothing to do
    if mapping_df.empty:
        return df

    # 2) Identify all unique search columns in the filtered mapping_df
    unique_search_cols = mapping_df['RoadsideColumnA'].dropna().unique()
    for col in unique_search_cols:
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].astype(str)

    # 3) Ensure all 'TransformedColumn' columns exist in df
    unique_replace_cols = mapping_df['TransformedColumn'].dropna().unique()
    for col in unique_replace_cols:
        if col not in df.columns:
            df[col] = None

    # 4) Group the mapping by (RoadsideColumnA, TransformedColumn)
    group_cols = ['RoadsideColumnA', 'TransformedColumn']
    grouped = mapping_df.groupby(group_cols, dropna=True)
    total_groups = grouped.ngroups

    # 5) We'll also build a mask for df's Roadway == 'T33'
    #    so we only modify those rows in the DataFrame.
    if 'Roadway' in df.columns:
        mask_roadway_df = (df['Roadway'] == 'T33')
    else:
        # If df doesn't have Roadway, we can't apply filtering, so do nothing
        return df

    # 6) Iterate with a progress bar
    for (col_search, col_replace), submap in tqdm(grouped,
                                                  total=total_groups,
                                                  desc="Step One Mapping Groups"):
        # Skip if df doesn’t actually have these columns
        if col_search not in df.columns or col_replace not in df.columns:
            continue

        replace_dict = {}
        null_val_replace = None

        # Build dictionary for all rows in this group
        for _, row in submap.iterrows():
            val_search = row['RoadsideValueA']
            val_replace = row['TransformedValue']

            if pd.isnull(val_search) or str(val_search).upper() == 'NULL':
                null_val_replace = val_replace
            else:
                replace_dict[str(val_search)] = val_replace

        # Single pass for null search
        if null_val_replace is not None:
            mask_null = mask_roadway_df & (df[col_search].isnull() | (df[col_search] == 'nan'))
            df.loc[mask_null, col_replace] = null_val_replace

        # Single pass for normal replacements
        mapped_series = df.loc[mask_roadway_df, col_search].map(replace_dict)  
        mask_mapped_notnull = mapped_series.notnull()

        # Assign only where we have a non-null mapped value
        df.loc[mask_roadway_df & mask_mapped_notnull, col_replace] = mapped_series[mask_mapped_notnull]

    # Replace literal string "None" with actual None
    df.replace('None', None, inplace=True)
    return df


def fetch_step_two_mapping(conn):
    """Fetch additional mapping data from SQL Server."""
    query = "SELECT * FROM NCTARECONPROD.dbo.tbMappingTableStep2"
    return pd.read_sql(query, conn)

def perform_step_two_mapping(df, mapping_df):
    """
    Step 2 Mapping, but only for rows where Roadway == 'T33' in the mapping table
    and also only modifies df rows where df['Roadway'] == 'T33'.
    """

    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    # 1) Filter the mapping table
    if 'Roadway' in mapping_df.columns:
        mapping_df = mapping_df[mapping_df['Roadway'] == 'T33']
    else:
        return df

    if mapping_df.empty:
        return df

    # 2) Cast df columns to string once if needed
    colA_list = mapping_df['RoadsideColumnA'].dropna().unique().tolist()
    colB_list = mapping_df['RoadsideColumnB'].dropna().unique().tolist()

    for cA in colA_list:
        if cA in df.columns and df[cA].dtype != object:
            df[cA] = df[cA].astype(str)

    for cB in colB_list:
        if cB in df.columns and df[cB].dtype != object:
            df[cB] = df[cB].astype(str)

    # 3) Ensure columns in 'TransformedColumn' exist
    for _, row in mapping_df.iterrows():
        trans_cols = row['TransformedColumn']
        if pd.notnull(trans_cols):
            for tcol in trans_cols.split(','):
                tcol = tcol.strip()
                if tcol not in df.columns:
                    df[tcol] = None

    # 4) We'll build a roadway mask for df
    if 'Roadway' not in df.columns:
        return df
    mask_roadway_df = (df['Roadway'] == 'T33')

    # 5) Use a progress bar around the main loop
    with tqdm(total=len(mapping_df), desc="Step Two Mapping (T33)", unit="row") as pbar:
        for _, row in mapping_df.iterrows():
            colA = row['RoadsideColumnA']
            valA = row['RoadsideValueA']
            colB = row['RoadsideColumnB']
            valB = row['RoadsideValueB']
            trans_cols = row['TransformedColumn']
            trans_vals = row['TransformedValue']

            if pd.isnull(colA) or pd.isnull(colB) or pd.isnull(trans_cols) or pd.isnull(trans_vals):
                pbar.update(1)
                continue

            # Build masks for colA, colB
            if isinstance(valA, str) and valA.upper() == 'NULL':
                maskA = df[colA].isnull()
            else:
                maskA = (df[colA] == str(valA))

            if isinstance(valB, str) and valB.upper() == 'NULL':
                maskB = df[colB].isnull()
            else:
                maskB = (df[colB] == str(valB))

            # Combine with roadway mask
            combined_mask = mask_roadway_df & maskA & maskB

            # Parse TransformedColumn, TransformedValue
            col_list = [c.strip() for c in trans_cols.split(',')]
            val_list = [v.strip() for v in trans_vals.split(',')]
            if len(col_list) != len(val_list):
                pbar.update(1)
                continue

            # Assign
            for cfinal, vfinal in zip(col_list, val_list):
                if isinstance(vfinal, str) and vfinal.upper() in ['NONE', 'NULL']:
                    vfinal = None
                df.loc[combined_mask, cfinal] = vfinal

            pbar.update(1)

    return df




def fetch_step_three_mapping(conn):
    """Fetch column renaming mapping data from SQL Server."""
    query = "SELECT ColumnA, ColumnB, ColumnC, FinalColumn FROM NCTARECONPROD.dbo.tbMappingOCRValues"
    return pd.read_sql(query, conn)

def perform_step_three_mapping(df, mapping_df):
    """
    Step 3 Mapping, only applying to rows where mapping_df['Roadway'] == 'T33'
    and df['Roadway'] == 'T33'.
    """

    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    # 1) Filter mapping_df to Roadway == 'T33'
    if 'Roadway' in mapping_df.columns:
        mapping_df = mapping_df[mapping_df['Roadway'] == 'T33']
    else:
        return df

    if mapping_df.empty:
        return df

    # 2) Cast relevant columns to string once
    all_cols = set(mapping_df['ColumnA'].dropna()).union(
        set(mapping_df['ColumnB'].dropna()),
        set(mapping_df['ColumnC'].dropna()),
        set(mapping_df['FinalColumn'].dropna())
    )
    for col in all_cols:
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].astype(str)

    # 3) Ensure final columns exist
    for final_col in mapping_df['FinalColumn'].dropna().unique():
        if final_col not in df.columns:
            df[final_col] = None

    # 4) We'll also build a roadway mask for df
    if 'Roadway' not in df.columns:
        return df
    mask_roadway_df = (df['Roadway'] == 'T33')

    # 5) Mapping pass with a progress bar
    with tqdm(total=len(mapping_df), desc="Step Three Mapping (T33)", unit="row") as pbar:
        for _, row in mapping_df.iterrows():
            colA = row['ColumnA']
            colB = row['ColumnB']
            colC = row['ColumnC']
            final_col = row['FinalColumn']

            # Skip if missing
            if pd.isnull(colA) or pd.isnull(colB) or pd.isnull(colC) or pd.isnull(final_col):
                pbar.update(1)
                continue

            # Step 1: If df[colA] notnull => final_col = colA
            maskA = mask_roadway_df & df[colA].notnull()
            df.loc[maskA, final_col] = df.loc[maskA, colA]

            # Step 2: If final_col is still null & df[colB].notnull => final_col = colB
            maskB = mask_roadway_df & df[final_col].isnull() & df[colB].notnull()
            df.loc[maskB, final_col] = df.loc[maskB, colB]

            # Step 3: If final_col is still null & df[colC].notnull => final_col = colC
            maskC = mask_roadway_df & df[final_col].isnull() & df[colC].notnull()
            df.loc[maskC, final_col] = df.loc[maskC, colC]

            pbar.update(1)

    return df



Base = declarative_base()

class TriexReconDaily(Base):
    __tablename__ = 'tbtriexrecondailydetail'
    TransactionID = Column(String, primary_key=True)
    UpdatedBy = Column(String)
    UpdatedTimeStamp = Column(DateTime)

if __name__ == "__main__":
    main()