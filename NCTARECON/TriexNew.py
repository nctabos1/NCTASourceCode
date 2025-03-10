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

LOG_FILE = "error_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),   # Log to file
        logging.StreamHandler()          # Log to console
    ]
)

# Global variable to store the last processed row details
last_processed_row = {'num_records_processed': 0, 'index_id': 0}
processing_complete = False  # Global flag to track processing completion


def signal_handler(file_name, engine, sig, frame):
    """Signal handler for SIGINT."""
    logging.info(f"KeyboardInterrupt (ID: {sig}) has been caught. Cleaning up...")
    if last_processed_row['num_records_processed'] > 0 and not processing_complete:
        update_checkpoint(engine, file_name, last_processed_row['num_records_processed'], last_processed_row['index_id'], 'unprocessed')
    logging.info("Cleanup complete. Exiting...")
    exit(0)

def atexit_handler():
    """Handler to update the checkpoint when the program exits."""
    if last_processed_row['num_records_processed'] > 0 and not processing_complete:
        update_checkpoint(engine, file_name, last_processed_row['num_records_processed'], last_processed_row['index_id'], 'unprocessed')
    logging.info("Program exited. Checkpoint updated.")

atexit.register(atexit_handler)

def setup_signal_handlers(file_name, engine):
    """Setup signal handlers for various interruptions."""
    def handler(sig, frame):
        logging.info(f"Signal {sig} received. Cleaning up...")
        if last_processed_row['num_records_processed'] > 0 and not processing_complete:
            update_checkpoint(engine, file_name, last_processed_row['num_records_processed'], last_processed_row['index_id'], 'unprocessed')
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
            parsed_unprocessed = []
            for row in all_unprocessed:
                try:
                    base_name = row.filename.replace('.zip', '')
                    date_str = base_name[-8:]  # last 8 chars, e.g. 20250117
                    file_date = datetime.strptime(date_str, "%Y%m%d").date()
                    parsed_unprocessed.append((file_date, row))
                except ValueError:
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
        logging.error(f"An error occurred while listing files on SFTP server: {e}", exc_info=True)
        return []

def connect_to_sftp():
    """Connect to the SFTP server and return the SFTP client."""
    try:
        transport = paramiko.Transport((SFTP_HOST, 22))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        return sftp, transport
    except Exception as e:
        logging.error(f"An error occurred while connecting to SFTP server: {e}", exc_info=True)
        return None, None

def update_transaction_status(df):
    """
    Update the TransactionStatus to 'Batched' where:
    - TransactionFileName is not null.
    """
    condition = df['TransactionFileName'].notna()
    df.loc[condition, 'TransactionStatus'] = 'Batched'
    return df


def update_mir_reject_status(df):
    """
    Update TransactionStatus = 'MIR Reject' where MIRReject is NOT NULL and not 'N'.
    """
    if 'MIRReject' in df.columns:
        condition = df['MIRReject'].notna() & (df['MIRReject'] != 'N')
        df.loc[condition, 'TransactionStatus'] = 'MIR Reject'
    return df


def update_unknown_status(df):
    """
    Update the TransactionStatus to 'Unknown' where:
    1. TransactionType is 'T' or 'V'
    2. TransactionDeliveryDateTime is '1900-01-01 00:00:00.000000' or NULL
    3. TransactionStatus is 'Batched'
    """
    #df['TransactionDeliveryDateTime'] = pd.to_datetime(df['TransactionDeliveryDateTime'], errors='coerce')

    condition_T = (
        (df['TransactionType'] == 'T') &
        ((df['TransactionDeliveryDateTime'] == pd.Timestamp('1900-01-01 00:00:00.000000')) | df['TransactionDeliveryDateTime'].isnull()) &
        (df['TransactionStatus'] == 'Batched')
    )
    condition_V = (
        (df['TransactionType'] == 'V') &
        ((df['TransactionDeliveryDateTime'] == pd.Timestamp('1900-01-01 00:00:00.000000')) | df['TransactionDeliveryDateTime'].isnull()) &
        (df['TransactionStatus'] == 'Batched')
    )

    df.loc[condition_T | condition_V, 'TransactionStatus'] = 'Unknown'
    return df

def normalize_mirreject(df):
    """
    Normalize the MIRReject column so that:
      - if it's already 'N', leave it alone
      - numeric strings like '0.0' => '0', '1.0' => '1'
      - invalid (non-numeric) => None
    """

    if 'MIRReject' not in df.columns:
        logging.warning("Warning: 'MIRReject' column not found in DataFrame.")
        return df

    # Step 1: Identify rows that are NOT 'N'
    mask_not_n = (df['MIRReject'] != 'N')

    # Step 2: Convert only those rows to numeric
    df.loc[mask_not_n, 'MIRReject'] = pd.to_numeric(
        df.loc[mask_not_n, 'MIRReject'],
        errors='coerce'
    )

    # Step 3: Convert numeric types back to integer strings or None
    df.loc[mask_not_n, 'MIRReject'] = df.loc[mask_not_n, 'MIRReject'].apply(
        lambda x: str(int(x)) if pd.notnull(x) else None
    )

    logging.info("\nDistinct MIRReject codes after normalization:")
    logging.info(df['MIRReject'].value_counts(dropna=False))

    return df


def download_csv_from_zip_on_sftp(sftp, sftp_path, local_path):
    """Download CSV file from within ZIP file on SFTP server to local path."""
    try:
        with sftp.open(sftp_path, 'rb') as zip_file:
            with zipfile.ZipFile(zip_file) as zip_ref:
                zip_file_list = zip_ref.namelist()
                csv_file_name = next((name for name in zip_file_list if name.endswith('.csv')), None)
                if not csv_file_name:
                    logging.error("No CSV file found in the ZIP archive.")
                    return None
                with zip_ref.open(csv_file_name) as csv_file:
                    local_file_path = os.path.join(local_path, os.path.basename(csv_file_name))
                    with open(local_file_path, 'wb') as local_file:
                        local_file.write(csv_file.read())
        return local_file_path

    except Exception as e:
        logging.error(f"An error occurred while extracting CSV file from ZIP archive on SFTP server: {e}", exc_info=True)
        return None

def load_csv_data(csv_path):
    """Load CSV data into a DataFrame and clean up 'None' strings."""
    df = pd.read_csv(csv_path, dtype=str)
    df.replace({'None': np.nan, '': np.nan}, inplace=True)
    return df

def rename_columns(conn, df):
    """
    Rename columns in DataFrame according to mapping table, 
    but skip renaming them in SQL because the DB already has the new names.
    """
    column_mapping_df = fetch_column_mapping(conn)
    column_mapping = {}
    
    for _, row in column_mapping_df.iterrows():
        old_column_name = row['RoadsideColumnName']
        new_column_name = row['TransformedColumnName']
        
        if old_column_name in df.columns:
            column_mapping[old_column_name] = new_column_name
            # Rename in the Pandas DataFrame
            df.rename(columns={old_column_name: new_column_name}, inplace=True)
            

    return column_mapping


def fetch_column_mapping(conn):
    """Fetch column renaming mapping data from SQL Server."""
    query = "SELECT RoadsideColumnName, TransformedColumnName FROM tbMappingTableColumns"
    return pd.read_sql(query, conn)

def fetch_step_one_mapping(conn):
    """Fetch mapping data from SQL Server."""
    query = "SELECT * FROM NCTARECONPROD.dbo.tbMappingTableStep1"
    return pd.read_sql(query, conn)

def perform_step_one_mapping(df, mapping_df):
    # Filter to T33 if needed
    if 'Roadway' in mapping_df.columns:
        mapping_df = mapping_df[mapping_df['Roadway'] == 'T33']

    if mapping_df.empty:
        return df

    # Convert any 'NULL' or 'None' to actual None in the mapping table
    mapping_df = mapping_df.replace({'NULL': None, 'None': None, '': None, np.nan: None})

    # Ensure columns exist in df
    unique_search_cols = mapping_df['RoadsideColumnA'].dropna().unique()
    for col in unique_search_cols:
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].astype(str)

    # Ensure all final columns exist in df
    unique_replace_cols = mapping_df['TransformedColumn'].dropna().unique()
    for col in unique_replace_cols:
        if col not in df.columns:
            df[col] = None

    group_cols = ['RoadsideColumnA', 'TransformedColumn']
    grouped = mapping_df.groupby(group_cols, dropna=True)

    for (col_search, col_replace), submap in tqdm(grouped, desc="Step One Mapping Groups"):
        if col_search not in df.columns or col_replace not in df.columns:
            continue
        replace_dict = {}
        null_val_replace = None

        for _, row in submap.iterrows():
            val_search = row['RoadsideValueA']
            val_replace = row['TransformedValue']

            if pd.isnull(val_search):
                null_val_replace = val_replace
            else:
                replace_dict[str(val_search)] = val_replace

        # Apply the dictionary mapping
        mapped_series = df[col_search].map(replace_dict)

        # Apply the null value mapping
        if null_val_replace is not None:
            null_mask = df[col_search].isnull() | (df[col_search] == 'nan')
            df.loc[null_mask, col_replace] = null_val_replace

        df[col_replace] = mapped_series.where(mapped_series.notnull(), df[col_replace])

    return df

def fetch_step_two_mapping(conn):
    """Fetch additional mapping data from SQL Server."""
    query = "SELECT * FROM NCTARECONPROD.dbo.tbMappingTableStep2"
    return pd.read_sql(query, conn)

def perform_step_two_mapping(df, mapping_df):
    if 'Roadway' in mapping_df.columns:
        mapping_df = mapping_df[mapping_df['Roadway'] == 'T33']
    else:
        return df

    if mapping_df.empty:
        return df

    # Convert 'NULL'/'None' to None
    mapping_df = mapping_df.replace({'NULL': None, 'None': None, '': None, np.nan: None})

    colA_list = mapping_df['RoadsideColumnA'].dropna().unique()
    colB_list = mapping_df['RoadsideColumnB'].dropna().unique()

    for col in colA_list:
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].astype(str)
    for col in colB_list:
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].astype(str)

    transformed_cols = mapping_df['TransformedColumn'].dropna().unique()
    for col in transformed_cols:
        if col not in df.columns:
            df[col] = None

    for _, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc="Step Two Mapping"):
        colA = row['RoadsideColumnA']
        valA = row['RoadsideValueA']
        colB = row['RoadsideColumnB']
        valB = row['RoadsideValueB']
        trans_col = row['TransformedColumn']
        trans_val = row['TransformedValue']

        if not colA or not colB or not trans_col or pd.isnull(trans_val):
            continue
        if (colA not in df.columns) or (colB not in df.columns) or (trans_col not in df.columns):
            continue

        # Build masks
        if pd.isnull(valA):
            maskA = df[colA].isnull() | (df[colA] == 'nan')
        else:
            maskA = df[colA] == str(valA)

        if pd.isnull(valB):
            maskB = df[colB].isnull() | (df[colB] == 'nan')
        else:
            maskB = df[colB] == str(valB)

        df.loc[maskA & maskB, trans_col] = trans_val

    return df

def fetch_step_three_mapping(conn):
    """Fetch column renaming mapping data from SQL Server."""
    query = "SELECT * FROM NCTARECONPROD.dbo.tbMappingOCRValues"
    return pd.read_sql(query, conn)

def perform_step_three_mapping(df, mapping_df):
    # Filter to T33 if that is relevant
    if 'Roadway' in mapping_df.columns:
        mapping_df = mapping_df[mapping_df['Roadway'] == 'T33']
    if mapping_df.empty:
        return df

    for _, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc="Step Three Mapping"):
        colA = row.get('ColumnA')
        colB = row.get('ColumnB')
        colC = row.get('ColumnC')
        final_col = row.get('FinalColumn')

        if not colA or not colB or not colC or not final_col:
            continue

        if colA not in df.columns:
            df[colA] = None
        if colB not in df.columns:
            df[colB] = None
        if colC not in df.columns:
            df[colC] = None
        if final_col not in df.columns:
            df[final_col] = None

        # If colA not null => final_col = colA
        maskA = df[colA].notnull()
        df.loc[maskA, final_col] = df.loc[maskA, colA]

        # If final_col still null => use colB
        maskB = df[final_col].isnull() & df[colB].notnull()
        df.loc[maskB, final_col] = df.loc[maskB, colB]

        # If final_col still null => use colC
        maskC = df[final_col].isnull() & df[colC].notnull()
        df.loc[maskC, final_col] = df.loc[maskC, colC]

    return df

def main():
    global engine, file_name, processing_complete, last_processed_row
    try:
        # 1) Create the database engine
        engine = create_engine(DATABASE_URI)

        while True:
            # 2) Connect to SFTP server
            sftp, transport = connect_to_sftp()
            if sftp is None or transport is None:
                logging.error("Failed to establish SFTP connection. Exiting.")
                break

            logging.info("Determining the next file to process...")

            # 3) Determine the next file to process
            file_name = get_next_file_to_process(engine, sftp, '/Dataingest/')
            if not file_name:
                logging.info("No more files to process.")
                sftp.close()
                transport.close()
                break

            logging.info(f"Processing file: {file_name}")

            # 4) Download the CSV from within the ZIP file on SFTP
            sftp_full_path = f'/Dataingest/{file_name}'
            local_path = r'R:\NCTAFileProcessing\LoadData\RoadsideCSVs'

            logging.info(f"Downloading CSV from within ZIP file from SFTP server: {sftp_full_path}")
            local_csv_path = download_csv_from_zip_on_sftp(sftp, sftp_full_path, local_path)

            if not local_csv_path:
                logging.error("Failed to download CSV file from ZIP on SFTP.")
                sftp.close()
                transport.close()
                break

            # 5) Load CSV data
            triex_recon_df = load_csv_data(local_csv_path)
            if triex_recon_df.empty:
                logging.error("CSV file could not be loaded or is empty.")
                sftp.close()
                transport.close()
                break

            # For debugging
            null_bos_delivery_time_count = triex_recon_df[
                triex_recon_df['BOS_DELIVERY_TIME'].isnull() & triex_recon_df['EFI_FILE_NAME'].notnull()
            ].shape[0]
            logging.info(f"Debug: BOS_DELIVERY_TIME null but EFI_FILE_NAME present count: {null_bos_delivery_time_count}")

            # 6) Normalize MIRReject
            triex_recon_df = normalize_mirreject(triex_recon_df)
            triex_recon_df.replace({'None': np.nan, '': np.nan}, inplace=True)

            # 7) Reflect DB metadata
            metadata = MetaData()
            metadata.reflect(bind=engine)

            with engine.connect() as conn:
                # 7a) Check existing checkpoint for partial progress
                checkpoint_table = Table('tbCheckpoint', metadata, autoload_with=engine)
                checkpoint_record = conn.execute(
                    select(checkpoint_table).where(checkpoint_table.c.filename == file_name)
                ).fetchone()

                last_processed_index = 0
                last_processed_id = 0
                if checkpoint_record:
                    last_processed_index = checkpoint_record.numrecordsprocessed or 0
                    last_processed_id = checkpoint_record.indexid or 0

                    if last_processed_index >= len(triex_recon_df):
                        logging.info("All rows in the file have already been processed.")
                        update_checkpoint(
                            engine,
                            file_name,
                            num_records_processed=last_processed_index,
                            index_id=last_processed_id,
                            status="processed"
                        )
                        sftp.close()
                        transport.close()
                        continue

                # Re-initialize the global last_processed_row
                last_processed_row['num_records_processed'] = last_processed_index
                last_processed_row['index_id'] = last_processed_id

                # Slice the DataFrame
                triex_recon_df = triex_recon_df.iloc[last_processed_index:].copy()

                # 7b) Rename columns
                rename_columns(conn, triex_recon_df)

                # 7c) Step One Mapping
                step_one_mapping_df = fetch_step_one_mapping(conn)
                triex_recon_df = perform_step_one_mapping(triex_recon_df, step_one_mapping_df)
                logging.info("Step one mapping complete")

                # 7d) Step Two Mapping
                step_two_mapping_df = fetch_step_two_mapping(conn)
                triex_recon_df = perform_step_two_mapping(triex_recon_df, step_two_mapping_df)
                logging.info("Step two mapping complete")

                # 7e) Step Three Mapping
                step_three_mapping_df = fetch_step_three_mapping(conn)
                triex_recon_df = perform_step_three_mapping(triex_recon_df, step_three_mapping_df)
                logging.info("Step three mapping complete")

                # 7f) Data Transformation and Status Updates
                triex_recon_df = update_mir_reject_status(triex_recon_df)
                triex_recon_df = update_unknown_status(triex_recon_df)
                triex_recon_df = update_transaction_status(triex_recon_df)

            # 8) Convert columns to string (except MIRReject) and replace 'nan'
            cols_to_exclude = ['MIRReject']
            for col in triex_recon_df.columns:
                if col not in cols_to_exclude:
                    triex_recon_df[col] = triex_recon_df[col].astype(str)
            triex_recon_df.replace('nan', None, inplace=True)

            # 9) Convert TransactionID safely to nullable integer
            triex_recon_df['TransactionID'] = pd.to_numeric(
                triex_recon_df['TransactionID'],
                errors='coerce'
            ).astype('Int64')

            # 10) Setup signal handlers
            setup_signal_handlers(file_name, engine)

            # 11) Insert transformed data into SQL Server with advanced duplicate handling
            insert_data_into_sql(engine, triex_recon_df, file_name)
            processing_complete = True

            # Close SFTP before next loop iteration
            sftp.close()
            transport.close()

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)


def update_checkpoint(engine, file_name, num_records_processed, index_id, status):
    """Update the tbCheckpoint table with the current processing status."""
    try:
        Session = sessionmaker(bind=engine)
        session = Session()
        metadata = MetaData()
        metadata.reflect(bind=engine)
        checkpoint_table = Table('tbCheckpoint', metadata, autoload_with=engine)
        
        current_timestamp = datetime.now()
        num_records_processed = int(num_records_processed)
        index_id = int(index_id)
        
        existing_record = session.execute(
            select(checkpoint_table).where(checkpoint_table.c.filename == file_name)
        ).fetchone()
        
        if existing_record:
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
        logging.error(f"Failed to update checkpoint: {e}", exc_info=True)
    finally:
        session.close()

def insert_data_into_sql(engine, df, file_name, batch_size=1000, max_retries=5, retry_wait=1):
    """
    Insert or update transformed data into SQL Server for TRIX using batch inserts.
    This version pre-checks each batch to separate new records from duplicates.
    """
    global last_processed_row
    try:
        # TRIX table names
        table_name = 'tbtriexrecondailydetail'
        history_table_name = 'tbtriexrecondailydetailhist'

        # Add audit/information columns
        current_user = 'NCTA_Reports'
        current_timestamp = datetime.now()
        df['UpdatedBy'] = current_user
        df['UpdatedTimeStamp'] = current_timestamp
        df['IngestName'] = file_name  # Track which file was ingested

        # --- Unified Datetime Conversion ---
        # Define the columns that are supposed to contain datetimes.
        datetime_columns = [
            'TransactionDateTime',
            'DispositionReceiptDateTime',
            'TransactionDeliveryDateTime',
            'TimeRecievedByRSS'
        ]

        for col in datetime_columns:
            if col in df.columns:
                # If the value is a string and longer than 26 characters, truncate to 26
                # (e.g., removing extremely long microsecond or nanosecond strings).
                df[col] = df[col].apply(
                    lambda x: x[:26] if isinstance(x, str) and len(x) > 26 else x
                )
                # Attempt to parse as datetime. We are *not* enforcing a single format string here,
                # since some columns might be in different date/time formats. If you know the exact 
                # format for all columns, you could pass `format="..."`.
                df[col] = pd.to_datetime(df[col], errors='coerce')
                # Reformat each datetime column as "YYYY-MM-DD HH:MM:SS" if not null
                df[col] = df[col].apply(
                    lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x) else None
                )

        # Replace stray 'nan' strings and np.nan with None
        df.replace(['nan', np.nan, 'None'], None, inplace=True)

        # Create a SQLAlchemy session and reflect the metadata
        Session = sessionmaker(bind=engine)
        session = Session()
        metadata = MetaData()
        metadata.reflect(bind=engine)
        table = Table(table_name, metadata, autoload_with=engine)
        history_table = Table(history_table_name, metadata, autoload_with=engine)

        total_rows = len(df)
        logging.info(f"Starting batch insert for {total_rows} rows.")
        with tqdm(total=total_rows, desc="Inserting Data", unit="row") as pbar:
            start_idx = 0
            while start_idx < total_rows:
                end_idx = start_idx + batch_size
                chunk_df = df.iloc[start_idx:end_idx]
                _ = insert_batch_with_duplicate_handling(
                        session,
                        table,
                        history_table,
                        chunk_df,
                        file_name,
                        engine,
                        max_retries,
                        retry_wait,
                        pbar
                    )
                start_idx += batch_size

        # Once all rows are processed, update the checkpoint as 'processed'
        if not df.empty:
            update_checkpoint(engine, file_name, len(df), df['TransactionID'].iloc[-1], 'processed')
        logging.info("Data inserted/updated successfully.")

    except IntegrityError as ie:
        logging.error(f"Integrity error: {ie}")
    except Exception as e:
        logging.error(f"Error inserting data: {e}", exc_info=True)
    finally:
        session.close()
        logging.info("Database session closed.")




def insert_batch_with_duplicate_handling(session, table, history_table, df_chunk, file_name, engine, max_retries=5, retry_wait=1, pbar=None):
    """
    Process a DataFrame chunk by first pre-checking which rows are new and which are duplicates.
    - New records are bulk inserted.
    - Duplicate records are processed row-by-row using duplicate-resolution logic.
    
    Before inserting, each record is checked for the mandatory column 'TransactionDeliveryDateTime'.
    If that value is None, it is replaced with the default "1900-01-01 00:00:00".
    
    Duplicate resolution now includes:
      - If the new row’s TransactionStatus is "Waiting MIR" and the existing row’s TransactionStatus is "Unknown",
        then the new row is given preference (after archiving the existing record) and updates the main table.
      - In all cases, if a duplicate does not update the main table, it is inserted into the history table.
    """
    from sqlalchemy import select
    from sqlalchemy.exc import IntegrityError, OperationalError
    global last_processed_row

    # Convert the chunk to a list of dictionaries.
    records = df_chunk.to_dict("records")
    if not records:
        return 0

    # --- Ensure mandatory column 'TransactionDeliveryDateTime' is not None ---
    # If the value is None, assign a default value.
    for r in records:
        if r.get("TransactionDeliveryDateTime") is None:
            r["TransactionDeliveryDateTime"] = "1900-01-01 00:00:00"

    # --- Pre-check: Partition new vs. duplicate records ---
    transaction_ids = [r["TransactionID"] for r in records]
    existing_query = select(table.c.TransactionID).where(table.c.TransactionID.in_(transaction_ids))
    existing_result = session.execute(existing_query).fetchall()
    existing_ids = {row[0] for row in existing_result}

    new_records = [r for r in records if r["TransactionID"] not in existing_ids]
    duplicate_records = [r for r in records if r["TransactionID"] in existing_ids]

    # --- Bulk insert new records using the proper transaction block ---
    if new_records:
        try:
            if session.in_transaction():
                # Already in a transaction: use a nested transaction (SAVEPOINT)
                with session.begin_nested():
                    session.execute(table.insert(), new_records)
            else:
                # No transaction active: start a new one
                with session.begin():
                    session.execute(table.insert(), new_records)
            for row_dict in new_records:
                last_processed_row["num_records_processed"] += 1
                last_processed_row["index_id"] = row_dict["TransactionID"]
                if pbar is not None:
                    pbar.update(1)
        except Exception as e:
            session.rollback()
            logging.error("Error during bulk insert of new records: %s", e)
            # Fallback: process new records row-by-row if needed.
            for row_dict in new_records:
                process_record_row_by_row(session, table, history_table, row_dict, max_retries, retry_wait, pbar)

    # --- Process duplicate records row-by-row ---
    for row_dict in duplicate_records:
        process_record_row_by_row(session, table, history_table, row_dict, max_retries, retry_wait, pbar)

    session.commit()
    return len(new_records) + len(duplicate_records)


def process_record_row_by_row(session, table, history_table, row_dict, max_retries, retry_wait, pbar):
    """
    Process a single record using duplicate-resolution logic with retries.
    This function:
      - Ensures the mandatory column 'TransactionDeliveryDateTime' is not None (assigns default if needed).
      - Retrieves the existing record.
      - Compares key date fields and TransactionStatus.
      - Applies the special condition: if new status is "Waiting MIR" and existing status is "Unknown",
        then update the main table.
      - Otherwise, if conditions for update are met (by date or by known status), update the main table
        (after archiving the existing record).
      - If none of the update conditions are met, inserts the row into the history table.
    """
    from sqlalchemy import select
    from sqlalchemy.exc import IntegrityError, OperationalError

    # Ensure mandatory column 'TransactionDeliveryDateTime' is not None.
    if row_dict.get("TransactionDeliveryDateTime") is None:
        row_dict["TransactionDeliveryDateTime"] = "1900-01-01 00:00:00"

    retries = 0
    while retries < max_retries:
        try:
            with session.begin_nested():
                # Fetch the existing record
                existing = session.execute(
                    select(table).where(table.c.TransactionID == row_dict["TransactionID"])
                ).fetchone()
                if existing:
                    existing_row_dict = dict(zip(table.columns.keys(), existing))
                    existing_disp_date = pd.to_datetime(existing_row_dict.get("DispositionReceiptDateTime"), errors="coerce")
                    new_disp_date = pd.to_datetime(row_dict.get("DispositionReceiptDateTime"), errors="coerce")
                    existing_trans_date = pd.to_datetime(existing_row_dict.get("TransactionDeliveryDateTime"), errors="coerce")
                    new_trans_date = pd.to_datetime(row_dict.get("TransactionDeliveryDateTime"), errors="coerce")
                    new_status = row_dict.get("TransactionStatus")
                    existing_status = existing_row_dict.get("TransactionStatus")

                    # Ensure we do not overwrite a non-null TransactionDeliveryDateTime with a null value
                    if existing_row_dict.get("TransactionDeliveryDateTime") is not None and row_dict.get("TransactionDeliveryDateTime") is None:
                        row_dict["TransactionDeliveryDateTime"] = existing_row_dict["TransactionDeliveryDateTime"]

                    # Refresh the UpdatedTimeStamp
                    row_dict["UpdatedTimeStamp"] = datetime.now()

                    # --- Duplicate resolution logic ---
                    if new_status == "Waiting MIR" and existing_status == "Unknown":
                        # Give preference to "Waiting MIR" by updating the main table.
                        session.execute(history_table.insert().values(existing_row_dict))
                        session.execute(
                            table.update()
                            .where(table.c.TransactionID == row_dict["TransactionID"])
                            .values(row_dict)
                        )
                    elif new_status in ["Batched", "MIR Reject", "Duplicate"]:
                        session.execute(history_table.insert().values(existing_row_dict))
                        session.execute(
                            table.update()
                            .where(table.c.TransactionID == row_dict["TransactionID"])
                            .values(row_dict)
                        )
                    elif all([
                        pd.isnull(existing_disp_date),
                        pd.isnull(existing_trans_date),
                        pd.isnull(new_disp_date),
                        pd.isnull(new_trans_date)
                    ]):
                        session.execute(history_table.insert().values(existing_row_dict))
                        session.execute(
                            table.update()
                            .where(table.c.TransactionID == row_dict["TransactionID"])
                            .values(row_dict)
                        )
                    elif (existing_disp_date is None) or (new_disp_date is not None and new_disp_date > existing_disp_date):
                        session.execute(history_table.insert().values(existing_row_dict))
                        session.execute(
                            table.update()
                            .where(table.c.TransactionID == row_dict["TransactionID"])
                            .values(row_dict)
                        )
                    elif (existing_trans_date is None) or (new_trans_date is not None and new_trans_date > existing_trans_date):
                        session.execute(history_table.insert().values(existing_row_dict))
                        session.execute(
                            table.update()
                            .where(table.c.TransactionID == row_dict["TransactionID"])
                            .values(row_dict)
                        )
                    else:
                        # In all other cases, insert the incoming row into history.
                        session.execute(history_table.insert().values(row_dict))
            break  # Break out of the retry loop if successful.
        except OperationalError as oe:
            if "deadlock victim" in str(oe):
                retries += 1
                logging.warning(f"Deadlock detected for TransactionID {row_dict['TransactionID']}. Retrying {retries}/{max_retries}...")
                session.rollback()
                time.sleep(retry_wait * (2 ** retries))
                continue
            else:
                session.rollback()
                raise
        except Exception as e:
            session.rollback()
            raise
    last_processed_row["num_records_processed"] += 1
    last_processed_row["index_id"] = row_dict["TransactionID"]
    if pbar is not None:
        pbar.update(1)








Base = declarative_base()

class TriexReconDaily(Base):
    __tablename__ = 'tbtriexrecondailydetail'
    TransactionID = Column(String, primary_key=True)
    UpdatedBy = Column(String)
    UpdatedTimeStamp = Column(DateTime)

if __name__ == "__main__":
    main()