from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import io
import time
from typing import Optional
from .utils import Config, logger, validate_file_size, normalize_column_names, optimize_dataframe
from .service import categorize_customer, categorize_by_due_date

router = APIRouter()

# IN-MEMORY DATA CACHE & PERSISTENCE
# ============================================================================
import os
import pickle

CACHE_FILE = "data_cache.pkl"
cached_dataframe = None

def save_cache(df):
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(df, f)
        logger.info(f"💾 Persistent cache saved to {CACHE_FILE}")
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                df = pickle.load(f)
            logger.info(f"📂 Persistent cache loaded from {CACHE_FILE}")
            return df
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    return None

@router.get("/")
def read_root():
    """Health check endpoint with system status."""
    return {
        "status": "running",
        "message": "Customer Categorization API - Production Ready",
        "version": "3.0",
        "max_file_size_mb": Config.MAX_FILE_SIZE_MB,
        "endpoints": {
            "unified_endpoint": "/data (supports file upload + query params)",
            "docs": "/docs",
            "health": "/"
        }
    }


@router.post("/data")
async def unified_data_endpoint(
    file: UploadFile = File(None),
    time_period: Optional[str] = None,
    include_details: bool = False
):
    """
    **UNIFIED ENDPOINT** - Handles both data upload and retrieval in a single endpoint.
    """
    global cached_dataframe
    
    start_time = time.time()
    logger.info(f"Unified endpoint called - File: {file.filename if file else 'None'}, Period: {time_period}")
    
    # ========================================================================
    # SCENARIO 1 & 4: File Upload
    # ========================================================================
    if file:
        logger.info(f"Processing file upload: {file.filename}")
        
        # Validate file type
        if not any(file.filename.endswith(ext) for ext in Config.ALLOWED_EXTENSIONS):
            logger.error(f"Invalid file type: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}"
            )
        
        # Validate file size
        if not validate_file_size(file):
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {Config.MAX_FILE_SIZE_MB}MB"
            )
        
        try:
            # Read uploaded file
            logger.info("Reading file contents...")
            contents = await file.read()
            
            if file.filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(contents))
                logger.info(f"Loaded CSV file with {len(df)} rows")
            else:
                df = pd.read_excel(io.BytesIO(contents), engine='openpyxl')
                logger.info(f"Loaded Excel file with {len(df)} rows")
            
            # Normalize column names
            df = normalize_column_names(df)
            logger.info(f"Columns found: {', '.join(df.columns.tolist())}")
            
            # Verify required columns
            required_columns = ['DUE_MONTH_2', 'DUE_MONTH_3', 'DUE_MONTH_4', 
                                'DUE_MONTH_5', 'DUE_MONTH_6', 'STATUS', 'LAST DUE REVD DATE']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing columns: {missing_columns}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required columns: {missing_columns}"
                )
            
            # Optimize dataframe
            df = optimize_dataframe(df)
            logger.info("Dataframe optimized for processing")
            
            # Apply categorizations
            logger.info("Applying payment history categorization...")
            df['Payment_Category'] = df.apply(categorize_customer, axis=1)
            
            logger.info("Applying due date categorization...")
            df['Due_Date_Category'] = df.apply(categorize_by_due_date, axis=1)
            
            # Cache the dataframe for subsequent requests
            cached_dataframe = df.copy()
            save_cache(cached_dataframe) # Save to disk
            logger.info("Data cached successfully for subsequent requests")
            logger.info(f"Cached dataframe shape: {cached_dataframe.shape}")
            logger.info(f"Cached columns: {cached_dataframe.columns.tolist()}")

            # SYNC TO MONGODB (For Reports View)
            logger.info("DEBUG: Starting MongoDB sync block...")
            try:
                logger.info("DEBUG: Importing get_call_data_collection...")
                from .db import get_call_data_collection
                
                logger.info("DEBUG: Getting collection...")
                collection = get_call_data_collection()
                
                logger.info("DEBUG: Converting dataframe to records...")
                records = df.to_dict('records')
                logger.info(f"DEBUG: Converted {len(records)} records.")
                
                # Handle NaN/None for MongoDB
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                            
                # Replace existing data
                logger.info("DEBUG: Deleting old data...")
                collection.delete_many({})
                
                if records:
                    logger.info("DEBUG: Inserting new data...")
                    collection.insert_many(records)
                    
                logger.info(f"✅ Synced {len(records)} records to MongoDB for Reports view")
            except Exception as e:
                logger.error(f"⚠️ Failed to sync to MongoDB: {e}", exc_info=True)

            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file: {str(e)}"
            )
    
    # ========================================================================
    # Check if we have data (or can load it from disk)
    # ========================================================================
    if cached_dataframe is None:
        cached_dataframe = load_cache()
    
    if cached_dataframe is None:
        logger.error("No cached data found and no file uploaded")
        raise HTTPException(
            status_code=400,
            detail="No data available. Please upload a file first."
        )
    
    df = cached_dataframe.copy()
    
    # Ensure numeric columns are numeric
    numeric_cols = ['ARREARS', 'AMOUNT', 'EMI']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # ========================================================================
    # Calculate KPIs
    # ========================================================================
    total_borrowers = len(df)
    logger.info(f"Calculating KPIs for {total_borrowers} borrowers")
    
    # Calculate total arrears
    if 'ARREARS' in df.columns:
        total_arrears = df['ARREARS'].sum()
    else:
        logger.warning("ARREARS column not found, returning 0")
        total_arrears = 0
    
    # Category counts
    payment_category_counts = {
        "consistent": len(df[df['Payment_Category'] == 'Consistent']),
        "inconsistent": len(df[df['Payment_Category'] == 'Inconsistent']),
        "overdue": len(df[df['Payment_Category'] == 'Overdue'])
    }
    
    due_date_category_counts = {
        "more_than_7_days": len(df[df['Due_Date_Category'] == 'More_than_7_days']),
        "1_to_7_days": len(df[df['Due_Date_Category'] == '1-7_days']),
        "today": len(df[df['Due_Date_Category'] == 'Today'])
    }
    
    # Base response
    response = {
        "status": "success",
        "kpis": {
            "total_borrowers": total_borrowers,
            "total_arrears": float(total_arrears),
            "by_payment_category": payment_category_counts,
            "by_due_date_category": due_date_category_counts
        },
        "data_cached": True,
        "file_uploaded": file is not None
    }
    
    # ========================================================================
    # Time period filter
    # ========================================================================
    if time_period:
        valid_periods = ["1-7_days", "More_than_7_days", "Today"]
        if time_period not in valid_periods:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid time_period. Must be one of: {', '.join(valid_periods)}"
            )
        
        logger.info(f"Filtering by time period: {time_period}")
        
        filtered_df = df[df['Due_Date_Category'] == time_period].copy()
        
        detail_columns = ['NO', 'BORROWER', 'AMOUNT', 'EMI', 'cell1', 'preferred_language']
        missing_detail_cols = [col for col in detail_columns if col not in df.columns]
        if missing_detail_cols:
            detail_columns = [col for col in detail_columns if col in df.columns]
        
        def get_color_indicator(payment_category):
            if payment_category == 'Consistent': return 'green'
            elif payment_category == 'Inconsistent': return 'orange'
            elif payment_category == 'Overdue': return 'red'
            return 'gray'
        
        filtered_df['indicator_color'] = filtered_df['Payment_Category'].apply(get_color_indicator)
        
        # Add additional useful columns if they exist
        for extra_col in ['TOTAL_LOAN', 'LAST_PAID_DATE', 'DUE_DATE']:
            if extra_col in df.columns:
                detail_columns.append(extra_col)

        result_columns = detail_columns + ['Payment_Category', 'indicator_color']
        borrowers_list = filtered_df[result_columns].to_dict('records')
        
        period_category_counts = {
            'consistent': len(filtered_df[filtered_df['Payment_Category'] == 'Consistent']),
            'inconsistent': len(filtered_df[filtered_df['Payment_Category'] == 'Inconsistent']),
            'overdue': len(filtered_df[filtered_df['Payment_Category'] == 'Overdue'])
        }
        
        response["period_filter"] = {
            "time_period": time_period,
            "total_borrowers_in_period": len(borrowers_list),
            "category_breakdown": period_category_counts,
            "borrowers": borrowers_list
        }
    
    # ========================================================================
    # Detailed breakdown
    # ========================================================================
    if include_details and not time_period:
        def get_color_indicator(payment_category):
            if payment_category == 'Consistent': return 'green'
            elif payment_category == 'Inconsistent': return 'orange'
            elif payment_category == 'Overdue': return 'red'
            return 'gray'
        
        if 'indicator_color' not in df.columns and 'Payment_Category' in df.columns:
            df['indicator_color'] = df['Payment_Category'].apply(get_color_indicator)
            
        detail_columns = ['NO', 'BORROWER', 'AMOUNT', 'EMI', 'cell1', 'Payment_Category', 'indicator_color', 'preferred_language', 'TOTAL_LOAN', 'LAST_PAID_DATE', 'DUE_DATE']
        # Filter for only existing columns
        available_detail_cols = [col for col in detail_columns if col in df.columns]
        
        logger.info(f"Available detail columns: {available_detail_cols}")
        
        response["detailed_breakdown"] = {
            "by_payment_category": {
                "Consistent": df[df['Payment_Category'] == 'Consistent'][available_detail_cols].to_dict('records'),
                "Inconsistent": df[df['Payment_Category'] == 'Inconsistent'][available_detail_cols].to_dict('records'),
                "Overdue": df[df['Payment_Category'] == 'Overdue'][available_detail_cols].to_dict('records')
            },
            "by_due_date_category": {
                "More_than_7_days": df[df['Due_Date_Category'] == 'More_than_7_days'][available_detail_cols].to_dict('records'),
                "1-7_days": df[df['Due_Date_Category'] == '1-7_days'][available_detail_cols].to_dict('records'),
                "Today": df[df['Due_Date_Category'] == 'Today'][available_detail_cols].to_dict('records')
            }
        }
    
    processing_time = time.time() - start_time
    response["processing_time_seconds"] = round(processing_time, 2)
    logger.info(f"Request completed successfully in {processing_time:.2f} seconds")
    
    return response


@router.get("/report_data")
def get_report_data():
    """
    Get specific columns for the Reports view from Local Cache (Parity with Dashboard).
    Columns: NO, BORROWER, AMOUNT, cell1, EMI, preferred_language, PAYMENT_CONFIRMATION, DATE, CALL_STATUS
    """
    try:
        # Fetch data from Local Cache
        global cached_dataframe
        if cached_dataframe is None:
            cached_dataframe = load_cache()
            
        if cached_dataframe is None:
            return {
                "status": "error",
                "message": "No data available. Please upload a file first using /upload endpoint.",
                "data": []
            }
        
        df = cached_dataframe.copy()
        
        # Define required columns and their mapping/defaults
        columns_config = [
            ("NO", ["NO", "No", "S.No"]),
            ("BORROWER", ["BORROWER", "Borrower", "Customer Name"]),
            ("AMOUNT", ["AMOUNT", "Amount", "Loan Amount"]),
            ("cell1", ["cell1", "Mobile", "Phone"]),
            ("EMI", ["EMI", "Emi"]),
            ("LAST_DUE_REVD_DATE", ["LAST DUE REVD DATE", "Last Due Revd Date", "LAST_DUE_REVD_DATE"]),
            ("FIRST_DUE_DATE", ["FIRST DUE DATE", "First Due Date", "FIRST_DUE_DATE"]),
            ("preferred_language", ["preferred_language", "Language"]),
            ("PAYMENT_CONFIRMATION", ["PAYMENT_CONFIRMATION", "Payment Confirmation"]),
            ("DATE", ["DATE", "Date", "FOLLOW_UP_DATE", "Follow Up Date"]),
            ("CALL_STATUS", ["CALL_STATUS", "Call Status", "Status"])
        ]
        
        result_data = []
        
        # Pre-calculate column mapping
        col_mapping = {}
        for out_col, candidates in columns_config:
            found_col = None
            for cand in candidates:
                if cand in df.columns:
                    found_col = cand
                    break
            col_mapping[out_col] = found_col

        # Helper to format dates
        def format_date_val(val):
            if pd.isna(val) or val is None or val == "":
                return ""
            try:
                # Try parsing as datetime
                dt = pd.to_datetime(val)
                return dt.strftime('%Y-%m-%d')
            except:
                # If parsing fails, just return as string (maybe clean "00:00:00")
                s = str(val)
                if "00:00:00" in s:
                    return s.replace("00:00:00", "").strip()
                return s

        # Iterate rows
        for _, row in df.iterrows():
            record = {}
            for out_col, src_col in col_mapping.items():
                if src_col:
                    val = row[src_col]
                    
                    # Special handling for Date columns
                    if out_col in ["LAST_DUE_REVD_DATE", "FIRST_DUE_DATE", "DATE", "PAYMENT_CONFIRMATION"]:
                        val = format_date_val(val)
                    else:
                        # Handle NaN/None for other cols
                        if pd.isna(val) or val is None:
                            val = ""
                        else:
                            val = str(val)
                else:
                    val = "" # Column not found
                
                record[out_col] = val
                
            result_data.append(record)
            
        return {
            "status": "success",
            "count": len(result_data),
            "data": result_data
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching data from Local Cache: {e}")
        return {
            "status": "error",
            "message": f"Error fetching data: {str(e)}",
            "data": []
        }

def update_call_status(borrower_id, call_status, payment_confirmation=None, follow_up_date=None):
    """
    Update call status and analysis results in the cached dataframe
    """
    global cached_dataframe
    if cached_dataframe is None:
        cached_dataframe = load_cache()
    
    if cached_dataframe is None:
        logger.warning(f"Cannot update call status for {borrower_id}: No data loaded")
        return False
        
    try:
        # Find row index for borrower
        # Assuming 'NO' is the ID column
        mask = cached_dataframe['NO'].astype(str) == str(borrower_id)
        if not mask.any():
            logger.warning(f"Borrower ID {borrower_id} not found in cache")
            return False
            
        idx = cached_dataframe.index[mask]
        
        # Update fields
        cached_dataframe.loc[idx, 'CALL_STATUS'] = call_status
        if payment_confirmation:
            cached_dataframe.loc[idx, 'PAYMENT_CONFIRMATION'] = payment_confirmation
        if follow_up_date:
            cached_dataframe.loc[idx, 'FOLLOW_UP_DATE'] = follow_up_date # Maps to DATE in report
            
        # Save back to cache
        save_cache(cached_dataframe)
        logger.info(f"✅ Updated status for {borrower_id}: {call_status} | {payment_confirmation}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating call status: {e}")
        return False



# ============================================================================
# NEW MONGODB UPLOAD ENDPOINT
# ============================================================================

from .db import get_call_data_collection
from datetime import datetime

@router.post("/upload")
async def upload_dataset_to_mongodb(file: UploadFile = File(None)):
    """
    Upload CSV/XLSX file, validate, parse, and store in MongoDB
    
    Purpose:
    - Upload CSV/XLSX
    - Validate file
    - Parse rows
    - Store data in MongoDB collection 'call_data'
    """
    start_time = time.time()
    logger.info(f"[UPLOAD] Received file upload request")
    
    if not file:
        logger.error("[ERROR] No file uploaded")
        raise HTTPException(
            status_code=400,
            detail="No file uploaded. Please provide a file with key 'file'."
        )

    logger.info(f"[UPLOAD] Processing file: {file.filename}")
    
    # Step 1: Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in Config.ALLOWED_EXTENSIONS:
        logger.error(f"[ERROR] Invalid file type: {file_ext}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {Config.ALLOWED_EXTENSIONS}"
        )
    
    # Step 2: Read file content
    try:
        contents = await file.read()
        file_size_kb = len(contents) / 1024
        logger.info(f"[FILE] File size: {file_size_kb:.2f} KB")
        
        # Validate file size (manual check since we have bytes)
        size_mb = file_size_kb / 1024
        if size_mb > Config.MAX_FILE_SIZE_MB:
            logger.error(f"[ERROR] File too large: {size_mb:.2f}MB (max: {Config.MAX_FILE_SIZE_MB}MB)")
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {Config.MAX_FILE_SIZE_MB}MB"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ERROR] Error reading file: {e}")
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    # Step 3: Parse file into DataFrame
    try:
        if file_ext == '.csv':
            df = pd.read_csv(io.BytesIO(contents))
        else:  # .xlsx or .xls
            df = pd.read_excel(io.BytesIO(contents))
        
        logger.info(f"[SUCCESS] File parsed successfully. Rows: {len(df)}, Columns: {len(df.columns)}")
        
    except Exception as e:
        logger.error(f"[ERROR] Error parsing file: {e}")
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")
    
    # Step 4: Normalize column names
    df = normalize_column_names(df)
    logger.info(f"[SUCCESS] Column names normalized")
    
    # Step 5: Convert DataFrame to list of records for MongoDB
    try:
        records = df.to_dict('records')
        
        # Add metadata to each record
        for record in records:
            record['_uploaded_at'] = datetime.utcnow()
            record['_source_file'] = file.filename
            
            # Convert any NaN values to None for MongoDB
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
        
        logger.info(f"[SUCCESS] Converted {len(records)} records for MongoDB")
        
    except Exception as e:
        logger.error(f"[ERROR] Error converting data: {e}")
        raise HTTPException(status_code=500, detail=f"Error converting data: {str(e)}")
    
    # Step 6: Store in MongoDB
    try:
        collection = get_call_data_collection()
        
        # Clear existing data and insert new
        collection.delete_many({})  # Remove old data
        result = collection.insert_many(records)
        
        logger.info(f"[SUCCESS] Successfully inserted {len(result.inserted_ids)} records into MongoDB")
        
    except Exception as e:
        logger.error(f"[ERROR] MongoDB error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    processing_time = time.time() - start_time
    
    return {
        "status": "success",
        "message": "File uploaded and stored in MongoDB successfully",
        "file_name": file.filename,
        "rows_inserted": len(records),
        "columns": list(df.columns),
        "processing_time": f"{processing_time:.2f}s"
    }

# ============================================================================
# NEW MONGODB FETCH ENDPOINT
# ============================================================================

@router.get("/customers")
async def get_customers(
    NO: Optional[str] = None,
    BORROWER: Optional[str] = None
):
    """
    Fetch customer data from MongoDB.
    
    Query Params:
    - NO: Filter by exact Borrower ID (NO column)
    - BORROWER: Filter by Borrower Name (partial match, case-insensitive)
    """
    try:
        collection = get_call_data_collection()
        query = {}
        
        # Build query based on parameters
        if NO:
            # Search for exact match on 'NO' column (stored as string or number in DB)
            # We try both just in case
            query["$or"] = [
                {"NO": NO},
                {"NO": int(NO)} if NO.isdigit() else {"NO": NO}
            ]
        elif BORROWER:
            # Case-insensitive regex search on 'BORROWER' column
            query["BORROWER"] = {"$regex": BORROWER, "$options": "i"}
            
        # Execute query
        # Exclude internal MongoDB _id field
        cursor = collection.find(query, {"_id": 0})
        
        # Convert to list
        results = list(cursor)
        
        return {
            "status": "success",
            "count": len(results),
            "filters": {
                "NO": NO,
                "BORROWER": BORROWER
            },
            "data": results
        }
        
    except Exception as e:
        logger.error(f"[ERROR] Error fetching customers: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )
