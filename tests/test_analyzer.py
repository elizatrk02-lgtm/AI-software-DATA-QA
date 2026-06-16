import pytest
import pandas as pd
import numpy as np
from src.analyzer import clean_data, check_pattern_errors

def test_clean_data_removes_duplicates():
    # Arrange: Create data with 1 explicit duplicate row
    raw_data = {
        'id':,
        'value': [10.0, 10.0, 20.0]
    }
    df = pd.DataFrame(raw_data)
    
    # Act: Process through cleaning engine
    cleaned_df = clean_data(df)
    
    # Assert: Should reduce rows from 3 down to 2
    assert len(cleaned_df) == 2
    assert cleaned_df.duplicated().sum() == 0

def test_clean_data_fills_gaps_with_median():
    # Arrange: Create list with a missing value (None)
    raw_data = {
        'score': [10, 20, None, 30, 40]  # Median of valid numbers is 25
    }
    df = pd.DataFrame(raw_data)
    
    # Act: Process
    cleaned_df = clean_data(df)
    
    # Assert: Missing value should now match the column median (25.0)
    assert cleaned_df['score'].isnull().sum() == 0
    assert cleaned_df['score'].iloc[2] == 25.0

def test_check_pattern_errors_catches_bad_emails():
    # Arrange: Create explicitly broken email strings
    raw_data = {
        'user_email': ['good@test.com', 'broken_email.com', 'admin@site.org']
    }
    df = pd.DataFrame(raw_data)
    
    # Act: Validate string format templates
    html_output = check_pattern_errors(df)
    
    # Assert: HTML output should explicitly flag the broken string
    assert "malformed email" in html_output
    assert "user_email" in html_output
