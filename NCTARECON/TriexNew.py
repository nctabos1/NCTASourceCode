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
    """Get the next file to process based on the checkpoint table and SFTP server."""
    with engine.connect() as conn:
        metadata = MetaData()
        checkpoint_table = Table('tbCheckpoint', metadata, autoload_with=engine)

        # Check for the last unprocessed "TRX" file
        unprocessed_record = conn.execute(
            select(checkpoint_table)
            .where(checkpoint_table.c.status == 'unprocessed')
            .where(checkpoint_table.c.filename.like('TRX%'))
            .order_by(checkpoint_table.c.CreatedDateTime.asc())
        ).fetchone()

        if unprocessed_record:
            logging.info(f"Found unprocessed file: {unprocessed_record.filename}")
            return unprocessed_record.filename

        # If all "TRX" files are processed, determine the next "TRX" file based on the last processed file date
        last_processed_record = conn.execute(
            select(checkpoint_table)
            .where(checkpoint_table.c.status == 'processed')
            .where(checkpoint_table.c.filename.like('TRX%'))
            .order_by(checkpoint_table.c.CreatedDateTime.desc())
        ).fetchone()

        if last_processed_record:
            last_processed_filename = last_processed_record.filename
            logging.info(f"Last processed file: {last_processed_filename}")
            last_processed_date = datetime.strptime(last_processed_filename.split('.')[0][-8:], "%Y%m%d")
            next_file_date = last_processed_date + timedelta(days=1)
            next_file_name = f'TRX{next_file_date.strftime("%Y%m%d")}.zip'
        else:
            # Start from the first expected date if no records exist
            next_file_date = datetime.strptime('20240729', "%Y%m%d")
            next_file_name = f'TRX{next_file_date.strftime("%Y%m%d")}.zip'
            logging.info(f"No processed files found. Starting from default date: {next_file_date}")

        # List files on the SFTP server
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
            logging.info(f"Next file to process: {next_file_name}")
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
    Update the TransactionStatus to 'Batched' where TransactionType is 'T'
    and TransactionDeliveryDateTime is not '1900-01-01 00:00:00'.
    """
    # Define the condition
    condition = (df['TransactionType'] == 'T') & (df['TransactionDeliveryDateTime'] != '1900-01-01 00:00:00')

    # Apply the update
    df.loc[condition, 'TransactionStatus'] = 'Batched'

    return df

def update_mir_reject_status(df):
    """
    Update the TransactionStatus to 'MIR Reject' where MIRReject is greater than or equal to 1.
    """
    # Convert 'MIRReject' column to float, ensuring NaNs are handled
    df['MIRReject'] = pd.to_numeric(df['MIRReject'], errors='coerce')
    
    # Convert to integer after handling NaNs
    df['MIRReject'] = df['MIRReject'].fillna(0).astype(int)
    
    # Define the condition
    condition = df['MIRReject'] >= 1

    # Apply the update
    df.loc[condition, 'TransactionStatus'] = 'MIR Reject'

    return df


def update_unknown_status(df):
    """
    Update the TransactionStatus to 'Unknown' where TransactionType is 'V',
    TransactionDeliveryDateTime is '1900-01-01 00:00:00', and TransactionStatus is 'Batched'.
    """
    # Define the condition
    condition = (
        (df['TransactionType'] == 'V') &
        (df['TransactionDeliveryDateTime'] == '1900-01-01 00:00:00') &
        (df['TransactionStatus'] == 'Batched')
    )

    # Apply the update
    df.loc[condition, 'TransactionStatus'] = 'Unknown'

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
                history_column_df = fetch_history_table(conn, metadata)
                rename_history_columns(conn, history_column_df)

                # Fetch step one mapping data from SQL Server
                step_one_mapping_df = fetch_step_one_mapping(conn)

                # Perform step one mapping
                triex_recon_df = perform_step_one_mapping(triex_recon_df, step_one_mapping_df)

                # Fetch step two mapping data from SQL Server
                step_two_mapping_df = fetch_step_two_mapping(conn)

                # Perform step two mapping
                triex_recon_df = perform_step_two_mapping(triex_recon_df, step_two_mapping_df)

                # Fetch Step 3 Mapping data from SQL Server
                step_three_mapping_df = fetch_step_three_mapping(conn)

                # Perform Step 3 mapping
                triex_recon_df = perform_step_three_mapping(triex_recon_df, step_three_mapping_df)

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




def insert_batch_with_duplicate_handling(session, table, history_table, batch, file_name, engine, max_retries=5, retry_wait=1, pbar=None):
    global last_processed_row
    rows_processed_since_last_checkpoint = 0

    for idx, row in batch.iterrows():
        retries = 0
        while retries < max_retries:
            try:
                # Start a new transaction if one is not already active
                if not session.is_active:
                    session.begin()

                # Attempt to insert the new row into the main table
                session.execute(table.insert().values(row.to_dict()))
                session.commit()
                
                break  # Break out of the retry loop if insert is successful
            except IntegrityError:
                # Handle duplicate TransactionID
                existing_row = session.execute(select(table).where(table.c.TransactionID == row['TransactionID'])).fetchone()
                
                if existing_row:
                    # Convert existing_row to a dictionary using indices
                    existing_row_dict = {column.name: value for column, value in zip(table.columns, existing_row)}

                    # Convert relevant fields to datetime if necessary
                    existing_disposition_date = pd.to_datetime(existing_row_dict.get('DispositionReceiptDateTime'), errors='coerce')
                    new_disposition_date = pd.to_datetime(row['DispositionReceiptDateTime'], errors='coerce')

                    existing_transaction_date = pd.to_datetime(existing_row_dict.get('TransactionDeliveryDateTime'), errors='coerce')
                    new_transaction_date = pd.to_datetime(row['TransactionDeliveryDateTime'], errors='coerce')

                    # Compare only if both dates are not None
                    if existing_disposition_date is None or (new_disposition_date is not None and new_disposition_date > existing_disposition_date):
                        # Insert the existing row into the history table
                        session.execute(history_table.insert().values(existing_row_dict))
                        
                        # Update the row in the main table with the new data
                        update_query = table.update().where(table.c.TransactionID == row['TransactionID']).values(row.to_dict())
                        session.execute(update_query)
                        session.commit()
                    else:
                        if existing_transaction_date is None or (new_transaction_date is not None and new_transaction_date > existing_transaction_date):
                            # Insert the existing row into the history table
                            session.execute(history_table.insert().values(existing_row_dict))
                            
                            # Update the row in the main table with the new data
                            update_query = table.update().where(table.c.TransactionID == row['TransactionID']).values(row.to_dict())
                            session.execute(update_query)
                            session.commit()
                        else:
                            # If the new transaction date is older, we do not update the main table but still add to history
                            history_query = history_table.insert().values(row.to_dict())
                            session.execute(history_query)
                            session.commit()
                break
            except OperationalError as oe:
                if "deadlock victim" in str(oe):
                    retries += 1
                    logging.warning(f"Deadlock detected. Retrying {retries}/{max_retries}...")
                    session.rollback()
                    time.sleep(retry_wait * (2 ** retries))  # Exponential backoff
                    continue
                else:
                    session.rollback()
                    raise
            except Exception as e:
                session.rollback()
                raise

        # Update checkpoint every 1000 rows
        rows_processed_since_last_checkpoint += 1
        if rows_processed_since_last_checkpoint >= 1000:
            update_checkpoint(engine, file_name, idx + 1, row['TransactionID'], 'unprocessed')
            rows_processed_since_last_checkpoint = 0

        # Update the global last processed row
        last_processed_row['num_records_processed'] = idx + 1
        last_processed_row['index_id'] = row['TransactionID']

        # Update the progress bar
        if pbar is not None:
            pbar.update(1)

    # Final checkpoint update
    if 'idx' in locals():
        update_checkpoint(engine, file_name, idx + 1, row['TransactionID'], 'unprocessed')



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
    df.replace({'None': np.nan, '': np.nan}, inplace=True)  # Replace 'None' and empty strings with NaN
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
    query = "SELECT * FROM NCTARECON.dbo.tbMappingTableStep1"
    return pd.read_sql(query, conn)

def perform_step_one_mapping(df, mapping_df):
    for _, row in tqdm(mapping_df.iterrows(), total=mapping_df.shape[0], desc="Step One Mapping"):
        try:
            value_to_replace = row['TransformedValue']
            column_to_search = row['RoadsideColumnA']
            current_value = row['RoadsideValueA']
            column_to_replace = row["TransformedColumn"]

            # Handle NULL/None transformations for TransformedValue
            if pd.isnull(value_to_replace) or value_to_replace in ['None', 'NULL']:
                value_to_replace = None

            # Check if current_value is null
            if pd.isnull(current_value) or current_value in ['None', 'NULL']:
                # Only perform mapping when the current value matches RoadsideValueA
                mask = df[column_to_search].isnull()
                df.loc[mask, column_to_replace] = value_to_replace
            else:
                if df[column_to_search].dtype != 'object':
                    df[column_to_search] = df[column_to_search].astype(str)
                
                # Only perform mapping when the current value matches RoadsideValueA
                df.loc[df[column_to_search] == str(current_value), column_to_replace] = value_to_replace

        except KeyError:
            continue

    # Replace 'None' strings with actual None to reflect NULL in SQL
    df.replace('None', None, inplace=True)
    
    return df

def fetch_step_two_mapping(conn):
    """Fetch additional mapping data from SQL Server."""
    query = "SELECT * FROM NCTARECON.dbo.tbMappingTableStep2"
    return pd.read_sql(query, conn)

def perform_step_two_mapping(df, mapping_df):
    for _, row in tqdm(mapping_df.iterrows(), total=mapping_df.shape[0], desc="Step Two Mapping"):
        try:
            if row['RoadsideColumnA'] in df.columns:
                filtered_rows = df[df[row['RoadsideColumnA']] == row['RoadsideValueA']]
                if pd.isnull(row['RoadsideValueB']):
                    filtered_rows = filtered_rows[pd.isnull(filtered_rows[row['RoadsideColumnB']])]
                else:
                    if pd.api.types.is_integer_dtype(df[row['RoadsideColumnB']]):
                        filtered_rows = filtered_rows[filtered_rows[row['RoadsideColumnB']].astype(str) == row['RoadsideValueB']]
                    else:
                        filtered_rows = filtered_rows[filtered_rows[row['RoadsideColumnB']] == row['RoadsideValueB']]

                if not filtered_rows.empty:
                    for final_column, final_value in zip(row['TransformedColumn'].split(','), row['TransformedValue'].split(',')):
                        df.loc[filtered_rows.index, final_column.strip()] = final_value.strip()

        except KeyError:
            continue

    return df

def fetch_step_three_mapping(conn):
    """Fetch column renaming mapping data from SQL Server."""
    query = "SELECT ColumnA, ColumnB, ColumnC, FinalColumn FROM NCTARECON.dbo.tbMappingOCRValues"
    return pd.read_sql(query, conn)

def perform_step_three_mapping(df, mapping_df):
    progress_bar = tqdm(total=len(mapping_df), desc="Performing Step Three Mapping", unit="step")
    
    for _, row in mapping_df.iterrows():
        try:
            columnA = row['ColumnA']
            columnB = row['ColumnB']
            columnC = row['ColumnC']
            final_column = row['FinalColumn']
            
            # Check if columnA exists and has a value
            if columnA in df.columns:
                mask = df[columnA].notnull()
                df.loc[mask, final_column] = df.loc[mask, columnA]
                continue  # Exit if a value is found in ColumnA
            
            # Check columnB if columnA is null
            if columnB in df.columns:
                mask = df[final_column].isnull() & df[columnB].notnull()
                df.loc[mask, final_column] = df.loc[mask, columnB]
                continue  # Exit if a value is found in ColumnB
            
            # Check columnC if both columnA and columnB are null
            if columnC in df.columns:
                mask = df[final_column].isnull() & df[columnB].isnull() & df[columnC].notnull()
                df.loc[mask, final_column] = df.loc[mask, columnC]
                
            progress_bar.update(1)
        except KeyError:
            continue

    progress_bar.close()

    return df

Base = declarative_base()

class TriexReconDaily(Base):
    __tablename__ = 'tbtriexrecondailydetail'
    TransactionID = Column(String, primary_key=True)
    UpdatedBy = Column(String)
    UpdatedTimeStamp = Column(DateTime)

if __name__ == "__main__":
    main()
