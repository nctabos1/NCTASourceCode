import pandas as pd
import datetime
import paramiko
from datetime import datetime
from tqdm import tqdm
import zipfile
import logging
import numpy as np
import os
from sqlalchemy import create_engine, MetaData, Table, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, DateTime
import time
import signal
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO)

# Database connection string
DATABASE_URI = 'mssql+pymssql://NCTA_Reports:NCTurnp!k3DB@report.prod.nctabos.net/NCTARECON'

# Global variable to store the last processed row details
last_processed_row = {'num_records_processed': 0, 'index_id': 0}

def signal_handler(file_name, engine, sig, frame):
    """Signal handler for SIGINT."""
    logging.info(f"KeyboardInterrupt (ID: {sig}) has been caught. Cleaning up...")
    if last_processed_row['num_records_processed'] > 0:
        update_checkpoint(engine, file_name, last_processed_row['num_records_processed'], last_processed_row['index_id'], 'interrupted')
    logging.info("Cleanup complete. Exiting...")
    exit(0)

def main():
    global engine, file_name
    try:
        # Load CSV data from SFTP server
        sftp_host = '10.225.16.6'
        sftp_username = 'MonroeDataIngest'
        sftp_password = 'f!%5mLsjf7'
        today_date = datetime.now().strftime("%Y%m%d")
        sftp_path = f'/DataIngest/ME_DI_20240710.zip'
        sftp_port = 22
        local_path = r'C:\Users\KhokharA\Documents\LoadData\RoadsideCSVs'  # Your local path

        # Download CSV file from SFTP server
        download_successful = download_csv_from_sftp(sftp_host, sftp_username, sftp_password, sftp_path, local_path)

        if not download_successful:
            logging.error("Failed to download CSV file from SFTP.")
            return

        # Load CSV Data from local path
        local_file_path = os.path.join(local_path, 'ME_DI_20240710.csv')
        file_name = os.path.basename(local_file_path)
        monroe_recon_df = load_csv_data(local_file_path)

        if monroe_recon_df is None or monroe_recon_df.empty:
            logging.error("CSV file could not be loaded or is empty.")
            return

        # Rename columns and get column mapping
        engine = create_engine(DATABASE_URI)
        metadata = MetaData()
        metadata.reflect(bind=engine)
        with engine.connect() as conn:
            rename_columns(conn, monroe_recon_df)

            # Rename history columns
            history_column_df = fetch_history_table(conn, metadata)
            rename_history_columns(conn, history_column_df)

            # Fetch step one mapping data from SQL Server
            step_one_mapping_df = fetch_step_one_mapping(conn)

            # Perform step one mapping
            monroe_recon_df = perform_step_one_mapping(monroe_recon_df, step_one_mapping_df)

            # Fetch step two mapping data from SQL Server
            step_two_mapping_df = fetch_step_two_mapping(conn)

            # Perform step two mapping
            monroe_recon_df = perform_step_two_mapping(monroe_recon_df, step_two_mapping_df)

            # Fetch Step 3 Mapping data from SQL Server
            step_three_mapping_df = fetch_step_three_mapping(conn)

            # Perform Step 3 mapping
            monroe_recon_df = perform_step_three_mapping(monroe_recon_df, step_three_mapping_df)

            # Round all decimal columns to 2 decimal places
            decimal_columns = ["AsignedFare", "FinancialDisposition"]
            monroe_recon_df[decimal_columns] = monroe_recon_df[decimal_columns].astype(float)
            monroe_recon_df[decimal_columns] = monroe_recon_df[decimal_columns].round(2)


            # Round float values to the nearest whole number without .0 and cast to string
            for col in monroe_recon_df.select_dtypes(include=[np.float64]).columns:
                monroe_recon_df[col] = monroe_recon_df[col].apply(lambda x: str(int(x)) if pd.notnull(x) and x.is_integer() else str(x))

            # Cast all values to strings
            monroe_recon_df = monroe_recon_df.astype(str)
            monroe_recon_df['TransactionID'] = monroe_recon_df['TransactionID'].astype(np.int64)

            # Replace 'nan' strings with None
            monroe_recon_df.replace('nan', None, inplace=True)

            # Truncate datetime columns to milliseconds precision
            datetime_columns = [
                'TransactionDateTime', 
                'DispositionReceiptDateTime', 
                'TimeRecievedByRSS', 
                'TransactionDeliveryDateTime'
            ]
            for col in datetime_columns:
                if col in monroe_recon_df.columns:
                    monroe_recon_df[col] = pd.to_datetime(monroe_recon_df[col], errors='coerce').dt.floor('ms')

            # Insert transformed data into SQL Server
            signal.signal(signal.SIGINT, partial(signal_handler, file_name, engine))
            insert_data_into_sql(engine, monroe_recon_df, file_name)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

def download_csv_from_sftp(sftp_host, sftp_username, sftp_password, sftp_path, local_path):
    """Download CSV file from SFTP server to local path."""
    try:
        transport = paramiko.Transport((sftp_host, 22))
        transport.connect(username=sftp_username, password=sftp_password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # Check if the file is a zip archive
        if sftp_path.endswith('.zip'):
            # Extract CSV from zip archive
            with sftp.open(sftp_path, 'rb') as file:
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    # Assuming there is only one CSV file in the zip
                    csv_filename = zip_ref.namelist()[0]
                    zip_ref.extract(csv_filename, local_path)
        else:
            # Directly download CSV file
            sftp.get(sftp_path, os.path.join(local_path, os.path.basename(sftp_path)))

        sftp.close()
        transport.close()

        return True

    except Exception as e:
        logging.error(f"An error occurred while downloading CSV file from SFTP server: {e}")
        return False

def load_csv_data(csv_path):
    """Load CSV data into a DataFrame."""
    return pd.read_csv(csv_path, dtype='str')

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
                alter_query = f"EXEC sp_rename 'tbmonroerecondetail.{old_column_name}', '{new_column_name}', 'COLUMN'"
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
    tb_history = Table('tbMonroeReconDetailHistory', metadata, autoload_with=conn)
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
                alter_query = f"EXEC sp_rename 'tbMonroeReconDetailHistory.{old_column_name}', '{new_column_name}', 'COLUMN'"
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

class MonroeReconDaily(Base):
    __tablename__ = 'tbmonroerecondetail'
    TransactionID = Column(String, primary_key=True)
    UpdatedBy = Column(String)
    UpdatedTimeStamp = Column(DateTime)

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
        logging.info(f"Checkpoint updated: {file_name}, {num_records_processed}, {index_id}, {status}")
    except Exception as e:
        logging.error(f"Failed to update checkpoint: {e}")
    finally:
        session.close()

def insert_data_into_sql(engine, df, file_name, batch_size=1000, max_retries=5, retry_wait=1):
    """Insert or update transformed data into SQL Server using bulk inserts and log updates to a history table."""
    global last_processed_row
    try:
        # Define the table names
        table_name = 'tbmonroerecondetail'
        history_table_name = 'tbMonroeReconDetailHistory'
        
        # Add UpdatedBy and UpdatedTimeStamp columns to the DataFrame
        current_user = 'NCTA_Reports'
        current_timestamp = datetime.now()
        df['UpdatedBy'] = current_user
        df['UpdatedTimeStamp'] = current_timestamp
        
        # Replace 'None' strings with actual None to reflect NULL in SQL
        df.replace('None', None, inplace=True)
        
        # Convert datetime columns to datetime and ensure they are within SQL Server's valid range
        df['TransactionDeliveryDateTime'] = pd.to_datetime(df['TransactionDeliveryDateTime'], errors='coerce')
        df['DispositionReceiptDateTime'] = pd.to_datetime(df['DispositionReceiptDateTime'], errors='coerce')
        
        # Replace out-of-range datetime values with a default valid datetime
        valid_datetime_range = pd.Timestamp('1900-01-01')
        df['TransactionDeliveryDateTime'] = df['TransactionDeliveryDateTime'].apply(
            lambda x: valid_datetime_range if pd.isna(x) or x < valid_datetime_range else x
        )
        df['DispositionReceiptDateTime'] = df['DispositionReceiptDateTime'].apply(
            lambda x: valid_datetime_range if pd.isna(x) or x < valid_datetime_range else x
        )
        
        # Create a SQLAlchemy session
        Session = sessionmaker(bind=engine)
        session = Session()
        metadata = MetaData()
        metadata.reflect(bind=engine)
        table = Table(table_name, metadata, autoload_with=engine)
        history_table = Table(history_table_name, metadata, autoload_with=engine)

        # Check existing checkpoint for the file
        checkpoint_record = session.execute(
            select(Table('tbCheckpoint', metadata, autoload_with=engine))
            .where(Table('tbCheckpoint', metadata, autoload_with=engine).c.filename == file_name)
        ).fetchone()

        last_processed_index = 0
        if checkpoint_record:
            last_processed_index = checkpoint_record.indexid + 2  # Start two rows after the last processed row

        if last_processed_index >= len(df):
            logging.info("New file detected. Processing from the beginning.")
            last_processed_index = 0

        df_to_process = df.iloc[last_processed_index:]

        # Process data row by row with a progress bar
        with tqdm(total=df_to_process.shape[0], desc="Inserting Data", unit="row") as pbar:
            insert_batch_with_duplicate_handling(session, table, history_table, df_to_process, file_name, engine, max_retries, retry_wait, pbar)

        if not df_to_process.empty:
            update_checkpoint(engine, file_name, len(df_to_process), df_to_process['TransactionID'].iloc[-1], 'processed')
        logging.info("Data inserted/updated successfully.")

    except IntegrityError as ie:
        logging.error(f"Integrity error: {ie}")
    except Exception as e:
        logging.error(f"Error inserting data: {e}")
    finally:
        session.close()
        logging.info("Database session closed.")

def insert_batch_with_duplicate_handling(session, table, history_table, batch, file_name, engine, max_retries=5, retry_wait=1, pbar=None):
    """Insert a batch of data into SQL Server, handling duplicates and updating the checkpoint after each row."""
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
                
                # Fetch the existing row from the main table
                existing_row = session.execute(select(table).where(table.c.TransactionID == row['TransactionID'])).fetchone()
                
                if existing_row:
                    # Convert existing_row to a dictionary using indices
                    existing_row_dict = {column.name: value for column, value in zip(table.columns, existing_row)}
                    
                    # Check if both DispositionReceiptDateTime and TransactionDeliveryDateTime are null
                    if (existing_row_dict.get('DispositionReceiptDateTime') is None and row['DispositionReceiptDateTime'] is None and 
                        existing_row_dict.get('TransactionDeliveryDateTime') is None and row['TransactionDeliveryDateTime'] is None):
                        
                        if row['TransactionStatus'] in ['Batched', 'Duplicate', 'MIR Reject']:
                            # Insert the existing row into the history table
                            session.execute(history_table.insert().values(existing_row_dict))
                            
                            # Update the row in the main table with the new data
                            update_query = table.update().where(table.c.TransactionID == row['TransactionID']).values(row.to_dict())
                            session.execute(update_query)
                            session.commit()
                        else:
                            # Directly add to the history table
                            history_query = history_table.insert().values(row.to_dict())
                            session.execute(history_query)
                            session.commit()
                    else:
                        # Compare based on DispositionReceiptDateTime first, then TransactionDeliveryDateTime
                        existing_disposition_date = existing_row_dict.get('DispositionReceiptDateTime')
                        new_disposition_date = row['DispositionReceiptDateTime']
                        
                        if existing_disposition_date is None or (new_disposition_date is not None and new_disposition_date > existing_disposition_date):
                            # Insert the existing row into the history table
                            session.execute(history_table.insert().values(existing_row_dict))
                            
                            # Update the row in the main table with the new data
                            update_query = table.update().where(table.c.TransactionID == row['TransactionID']).values(row.to_dict())
                            session.execute(update_query)
                            session.commit()
                        else:
                            # If DispositionReceiptDateTime is not relevant or both are None, compare TransactionDeliveryDateTime
                            existing_transaction_date = existing_row_dict.get('TransactionDeliveryDateTime')
                            new_transaction_date = row['TransactionDeliveryDateTime']
                            
                            if existing_transaction_date is None or new_transaction_date > existing_transaction_date:
                                # Insert the existing row into the history table
                                session.execute(history_table.insert().values(existing_row_dict))
                                
                                # Update the row in the main table with the new data
                                update_query = table.update().where(table.c.TransactionID == row['TransactionID']).values(row.to_dict())
                                session.execute(update_query)
                                session.commit()
                            elif (existing_disposition_date == new_disposition_date and existing_transaction_date == new_transaction_date):
                                # If both dates are the same, check TransactionStatus
                                if row['TransactionStatus'] in ['Batched', 'Duplicate', 'MIR Reject']:
                                    # Insert the existing row into the history table
                                    session.execute(history_table.insert().values(existing_row_dict))
                                    
                                    # Update the row in the main table with the new data
                                    update_query = table.update().where(table.c.TransactionID == row['TransactionID']).values(row.to_dict())
                                    session.execute(update_query)
                                    session.commit()
                                else:
                                    # Directly add to the history table
                                    history_query = history_table.insert().values(row.to_dict())
                                    session.execute(history_query)
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

# Call the function in your main workflow
if __name__ == "__main__":
    main()
