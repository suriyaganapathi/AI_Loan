import pandas as pd
from datetime import datetime, timedelta

def categorize_customer(row):
    """
    Categorize customer based on payment history.
    Returns: Consistent, Inconsistent, or Overdue
    """
    # Define payment month columns
    due_months = ['DUE_MONTH_2', 'DUE_MONTH_3', 'DUE_MONTH_4', 'DUE_MONTH_5', 'DUE_MONTH_6']
    
    # Count paid months
    paid_months = sum(1 for month in due_months if pd.notna(row.get(month)) and str(row.get(month)).strip())
    
    # Calculate missed months
    missed_months = 5 - paid_months
    
    # Get status
    status = str(row.get('STATUS', '')).upper().strip()
    
    # Apply categorization rules (ORDER MATTERS!)
    # Check Overdue first (NPA with low payments)
    if paid_months <= 3 and status == 'NPA':
        return 'Overdue'
    # Then check Inconsistent (missed many payments)
    elif missed_months >= 2:
        return 'Inconsistent'
    # Then check Consistent (good payment history with STD status)
    elif missed_months < 2 and status == 'STD':
        return 'Consistent'
    else:
        # Default to Inconsistent if doesn't match other rules
        return 'Inconsistent'


def categorize_by_due_date(row):
    """
    Categorize customer based on days until due date.
    Due date = LAST DUE REVD DATE + 30 days
    Returns: More_than_7_days, 1-7_days, or Today
    """
    try:
        # Get last payment date
        raw_date = row.get('LAST DUE REVD DATE')
        if pd.isna(raw_date):
            return 'Unknown'
        
        # Convert to datetime (handle if already datetime or needs parsing)
        if isinstance(raw_date, (datetime, pd.Timestamp)):
            last_due_date = raw_date
        else:
            last_due_date = pd.to_datetime(raw_date, dayfirst=True, errors='coerce')
        
        if pd.isna(last_due_date):
            return 'Date_Format_Error'
        
        # Calculate due date (last payment + 30 days)
        due_date = last_due_date + timedelta(days=30)
        
        # Get current date (without time)
        current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Calculate days until due
        days_left = (due_date - current_date).days
        
        # Categorize
        if days_left > 7:
            return 'More_than_7_days'
        elif 1 <= days_left <= 7:
            return '1-7_days'
        else:
            # 0 or negative (today or overdue)
            return 'Today'
    except Exception as e:
        return 'Parse_Error'
