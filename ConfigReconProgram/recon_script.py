import json
import os
import logging
import paramiko
import zipfile
import io
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, MetaData, Table, select, insert
import subprocess
import pymssql

# -----------------------------------------------------------------------------
# 1. Load configuration and set up logging using GitHub
# -----------------------------------------------------------------------------
def get_git_branch():
    """Returns the current git branch name."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception as e:
        print(f"Error determining git branch: {e}")
        return "unknown"

def get_roadway():
    """Determines the roadway from an environment variable."""
    roadway = os.getenv("ROADWAY", "").lower()
    if roadway not in ["triex", "monroe"]:
        raise ValueError("Invalid or missing roadway. Must be 'triex' or 'monroe'. Set the ROADWAY environment variable.")
    return roadway

def load_config():
    """Loads the appropriate config file based on Git branch and roadway."""
    branch = get_git_branch()
    roadway = os.getenv("ROADWAY", "").lower()

    if roadway not in ["triex", "monroe"]:
        raise ValueError(f"Invalid or missing ROADWAY environment variable. Must be 'triex' or 'monroe'. Current: {roadway}")

    env = "prod" if branch == "main" else "dev"
    config_filename = f"recon_config_{roadway}.{env}.json"

    print(f"Loading config file: {config_filename}")  # Debugging output

    if not os.path.exists(config_filename):
        raise FileNotFoundError(f"Config file {config_filename} not found.")

    with open(config_filename, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            raise ValueError(f"Config file {config_filename} is empty!")
        
        return json.loads(content)


# Load config
config = load_config()
    
# -----------------------------------------------------------------------------
# 2. Extract configurations and create DB engine
# -----------------------------------------------------------------------------
DATABASE_URI    = config["environment"]["DATABASE_URI"]
SFTP_HOST       = config["environment"]["SFTP_HOST"]
SFTP_USERNAME   = config["environment"]["SFTP_USERNAME"]
SFTP_PASSWORD   = config["environment"]["SFTP_PASSWORD"]

SFTP_DIRECTORY  = config["file_processing"]["sftp_directory"]
LOCAL_DIRECTORY = config["file_processing"]["local_directory"]
FILE_PREFIX     = config["file_processing"]["file_prefix"]
FILE_EXTENSION  = config["file_processing"]["file_extension"]
CHECKPOINT_TABLE = config["file_processing"]["checkpoint_table"]
MAIN_TABLE      = config["file_processing"]["main_table"]

engine = create_engine(DATABASE_URI)

# -----------------------------------------------------------------------------
# 3. Connect to SFTP
# -----------------------------------------------------------------------------
def connect_to_sftp():
    try:
        transport = paramiko.Transport((SFTP_HOST, 22))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        logging.info("Connected to SFTP successfully.")
        return sftp, transport
    except Exception as e:
        logging.error(f"Error connecting to SFTP: {e}", exc_info=True)
        return None, None

def sftp_file_exists(sftp, remote_path):
    try:
        sftp.stat(remote_path)
        return True
    except IOError:
        return False

# -----------------------------------------------------------------------------
# 4. Determine the next file to process based on the checkpoint table
# -----------------------------------------------------------------------------
def get_file_to_process(conn, sftp):
    try:
        metadata = MetaData()
        checkpoint_table = Table(CHECKPOINT_TABLE, metadata, autoload_with=conn)
        query_unprocessed = (
            select(checkpoint_table)
            .where(checkpoint_table.c.status == 'unprocessed')
            .where(checkpoint_table.c.filename.like(f"{FILE_PREFIX}%"))
        )
        all_unprocessed = conn.execute(query_unprocessed).fetchall()
        if all_unprocessed:
            parsed = []
            for row in all_unprocessed:
                try:
                    filename = row.filename.replace(FILE_EXTENSION, "")
                    date_str = filename[-8:]  # Expecting YYYYMMDD
                    file_date = datetime.strptime(date_str, "%Y%m%d").date()
                    parsed.append((file_date, row))
                except ValueError:
                    continue
            parsed.sort(key=lambda x: x[0])
            oldest_date, oldest_row = parsed[0]
            logging.info(f"Found unprocessed file: {oldest_row.filename}")
            return oldest_row.filename, oldest_row.numrecordsprocessed

        query_last = (
            select(checkpoint_table)
            .where(checkpoint_table.c.status == 'processed')
            .where(checkpoint_table.c.filename.like(f"{FILE_PREFIX}%"))
            .order_by(checkpoint_table.c.CreatedDateTime.desc())
        )
        last_record = conn.execute(query_last).fetchone()
        if last_record:
            last_filename = last_record.filename
            logging.info(f"No unprocessed file. Most recent processed: {last_filename}")
            try:
                last_date = datetime.strptime(last_filename.split('.')[0][-8:], "%Y%m%d")
            except ValueError:
                logging.error(f"Error parsing date from {last_filename}")
                return None, None
            next_date = last_date + timedelta(days=1)
            next_filename = f"{FILE_PREFIX}{next_date.strftime('%Y%m%d')}{FILE_EXTENSION}"
        else:
            next_date = datetime.strptime("20240729", "%Y%m%d")
            next_filename = f"{FILE_PREFIX}{next_date.strftime('%Y%m%d')}{FILE_EXTENSION}"
            logging.info(f"No records found. Starting from default date: {next_date}")

        remote_path = f"{SFTP_DIRECTORY.rstrip('/')}/{next_filename}"
        if sftp_file_exists(sftp, remote_path):
            conn.execute(
                insert(checkpoint_table).values(
                    filename=next_filename,
                    numrecordsprocessed=0,
                    indexid=0,
                    status='unprocessed',
                    CreatedDateTime=datetime.now(),
                    UpdatedDateTime=datetime.now()
                )
            )
            logging.info(f"Next file to process: {next_filename}")
            return next_filename, 0
        else:
            logging.info(f"File {next_filename} not found on SFTP.")
            return None, None

    except Exception as e:
        logging.error(f"Error in get_file_to_process: {e}", exc_info=True)
        return None, None

# -----------------------------------------------------------------------------
# 5. Download the ZIP from SFTP, extract the CSV, and save it locally
# -----------------------------------------------------------------------------
def extract_csv_from_sftp(sftp, file_name):
    sftp_zip_path = os.path.join(SFTP_DIRECTORY, file_name)
    try:
        with sftp.open(sftp_zip_path, "rb") as zstream:
            zip_data = zstream.read()
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if not csv_files:
                    logging.error(f"No CSV found in {file_name}.")
                    return None
                csv_file = csv_files[0]
                extracted_path = os.path.join(LOCAL_DIRECTORY, csv_file)
                with zip_ref.open(csv_file) as csvf, open(extracted_path, 'wb') as outf:
                    outf.write(csvf.read())
                logging.info(f"Extracted CSV {csv_file} to {extracted_path}")
                return extracted_path
    except Exception as e:
        logging.error(f"Error extracting CSV from {file_name}: {e}", exc_info=True)
        return None

# -----------------------------------------------------------------------------
# 6. Load the CSV into a pandas DataFrame
# -----------------------------------------------------------------------------
def load_csv_to_dataframe(csv_path, skip_rows=0):
    try:
        if skip_rows > 0:
            df = pd.read_csv(csv_path, skiprows=range(1, skip_rows+1), dtype=str)
            logging.info(f"Loaded CSV with {len(df)} rows after skipping {skip_rows} rows.")
        else:
            df = pd.read_csv(csv_path, dtype=str)
            logging.info(f"Loaded CSV with {len(df)} rows.")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV: {e}", exc_info=True)
        return None

# -----------------------------------------------------------------------------
# 7A. Rename Columns Function
# Uses only RoadsideColumnA and TransformedColumn.
# -----------------------------------------------------------------------------
def rename_columns_from_mapping(df, mapping_df, mapping_config):
    # Look for the mapping step with type "rename"
    rename_step = next((step for step in mapping_config['steps'] if step.get('type') == 'rename'), None)
    if not rename_step:
        logging.warning("No renaming step found in mapping configuration. Skipping renaming.")
        return df
    filters = rename_step.get('filter', {})
    filtered_mapping_df = mapping_df.copy()
    for key, value in filters.items():
        filtered_mapping_df = filtered_mapping_df[filtered_mapping_df[key] == value]
    if filtered_mapping_df.empty:
        logging.warning(f"No matching columns found for renaming using filters: {filters}. Skipping renaming.")
        return df
    rename_dict = dict(zip(filtered_mapping_df['RoadsideColumnA'], filtered_mapping_df['TransformedColumn']))
    logging.info(f"Renaming columns using filters {filters}: {rename_dict}")
    df.rename(columns=rename_dict, inplace=True)
    return df

# -----------------------------------------------------------------------------
# Helper: Apply an operator to a pandas Series
# -----------------------------------------------------------------------------
def apply_operator(series, operator, value):
    if operator == "==":
        return series == value
    elif operator == "!=" or operator.lower() == "is_not":
        return series != value
    elif operator == "is_not_null":
        return series.notnull() & (series != 'nan')
    elif operator == "is_null":
        return series.isnull()
    else:
        return series == value  # Default fallback


# -----------------------------------------------------------------------------
# 7B. Generic Mapping Function
# -----------------------------------------------------------------------------
def perform_generic_mapping(df, mapping_df):
    """
    Applies mapping rules from `mapping_df` to `df` based on conditions.
    Supports operators: '==', '!=', 'is_not_null', and 'is_null'.
    """
    for _, row in mapping_df.iterrows():
        op = row["Operator"] if pd.notnull(row["Operator"]) else "=="  # Default to '=='
        op_lower = op.lower() if pd.notnull(op) else None
        
        mask = pd.Series(True, index=df.index)  # Start with all True
        
        for col in row.index:
            if col.startswith("RoadsideColumn") and pd.notnull(row[col]):
                suffix = col.replace("RoadsideColumn", "")
                value_key = "RoadsideValue" + suffix
                expected_val = row.get(value_key)
                src_col = row[col]
                
                if src_col in df.columns:
                    # Handle NULL comparisons
                    if op_lower == "is_null":
                        mask &= df[src_col].isnull()
                    elif op_lower == "is_not_null":
                        mask &= df[src_col].notnull()
                    elif pd.isnull(expected_val):  # If RoadsideValueA is NULL
                        mask &= df[src_col].isnull()
                    elif isinstance(expected_val, str) and "1900-01-01" in expected_val:
                        # Normalize datetime format for comparison
                        df[src_col] = pd.to_datetime(df[src_col], errors='coerce')
                        expected_val = pd.to_datetime(expected_val)
                        mask &= (df[src_col] == expected_val)
                    else:
                        mask &= apply_operator(df[src_col], op, expected_val)
                else:
                    mask &= False  # If column not in DF, condition should fail
            
        target_col = row.get("TransformedColumn")
        target_val = row.get("TransformedValue")
        
        if pd.notnull(target_col):
            df.loc[mask, target_col] = target_val

    return df



# -----------------------------------------------------------------------------
# 7C. OCR Mapping Function
# -----------------------------------------------------------------------------
def perform_ocr_mapping(df, mapping_df):
    """
    For each mapping rule row, consider the source columns in order (RoadsideColumnA, then B, then C)
    and set the target column (TransformedColumn) to the first non-null value found.
    """
    for _, row in mapping_df.iterrows():
        src_cols = []
        if pd.notnull(row["RoadsideColumnA"]):
            src_cols.append(row["RoadsideColumnA"])
        if pd.notnull(row["RoadsideColumnB"]):
            src_cols.append(row["RoadsideColumnB"])
        if pd.notnull(row["RoadsideColumnC"]):
            src_cols.append(row["RoadsideColumnC"])
        final_col = row["TransformedColumn"]
        if not src_cols or pd.isnull(final_col):
            continue
        if final_col not in df.columns:
            df[final_col] = None
        for col in src_cols:
            if col in df.columns:
                mask = df[final_col].isnull() & df[col].notnull()
                df.loc[mask, final_col] = df.loc[mask, col]
    return df

# -----------------------------------------------------------------------------
# 8. Fetch mapping data using the master table and a filter dictionary.
# Now only filter by Roadway and Step.
# -----------------------------------------------------------------------------
def fetch_mapping_data(conn, mapping_table, filter_dict):
    query = f"SELECT * FROM {mapping_table}"
    if filter_dict:
        conditions = []
        for k, v in filter_dict.items():
            conditions.append(f"{k} = '{v}'")
        query += " WHERE " + " AND ".join(conditions)
    logging.info(f"Mapping query: {query}")
    return pd.read_sql(query, conn)

# -----------------------------------------------------------------------------
# 9. Apply mapping steps according to the config file and "Step" column.
# -----------------------------------------------------------------------------
def apply_mapping_steps(conn, df, mapping_config):
    mapping_table = mapping_config.get("table")
    steps = mapping_config.get("steps", [])
    for step in sorted(steps, key=lambda x: x["step"]):
        filter_dict = step.get("filter", {}).copy()
        # Add the Step value to the filter so only rules for that step are selected.
        filter_dict["Step"] = str(step["step"])
        mapping_df = fetch_mapping_data(conn, mapping_table, filter_dict)
        if mapping_df.empty:
            logging.info(f"No mapping data found for step {step['step']}. Skipping.")
            continue
        logging.info(f"Applying mapping step {step['step']}: {step.get('name')}")
        logging.debug(f"Mapping data (head):\n{mapping_df.head(3).to_string()}")
        logging.info(f"DataFrame head before step {step['step']}:\n{df.head(5).to_string()}")

        if step.get("type", "") == "rename":
            df = rename_columns_from_mapping(df, mapping_df, mapping_config)
        elif step.get("type", "") == "ocr":
            df = perform_ocr_mapping(df, mapping_df)
        else:
            df = perform_generic_mapping(df, mapping_df)

        logging.info(f"DataFrame head after step {step['step']}:\n{df.head(5).to_string()}")
    return df

# -----------------------------------------------------------------------------
# 10. Main process: Connect to SFTP/DB, process file, apply mapping, and output CSV.
# -----------------------------------------------------------------------------
def main():
    sftp, transport = connect_to_sftp()
    if not sftp:
        logging.error("SFTP connection failed. Exiting process.")
        return

    with engine.connect() as conn:
        file_name, records_processed = get_file_to_process(conn, sftp)
        if not file_name:
            logging.info("No file to process found. Exiting process.")
            sftp.close()
            transport.close()
            return
        logging.info(f"Processing file: {file_name}")

        local_csv_path = extract_csv_from_sftp(sftp, file_name)
        if not local_csv_path:
            logging.error("CSV extraction failed. Exiting process.")
            sftp.close()
            transport.close()
            return

        df = load_csv_to_dataframe(local_csv_path, skip_rows=records_processed)
        if df is None:
            logging.error("Failed to load CSV into DataFrame. Exiting process.")
            sftp.close()
            transport.close()
            return

        mapping_config = config.get("mapping", {})
        df = apply_mapping_steps(conn, df, mapping_config)

        base_name = os.path.basename(file_name).replace(FILE_EXTENSION, "")
        output_csv = os.path.join(LOCAL_DIRECTORY, f"transformed_{base_name}.csv")
        try:
            df.to_csv(output_csv, index=False)
            logging.info(f"Transformed DataFrame saved to {output_csv}")
            print(f"Transformed DataFrame saved to {output_csv}")
        except Exception as e:
            logging.error(f"Error saving transformed CSV: {e}", exc_info=True)

    sftp.close()
    transport.close()
    logging.info("Process completed.")

if __name__ == "__main__":
    main()
