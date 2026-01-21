from fpdf import FPDF
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import json
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import PipelineLogger

class DataValidator:
    """Validates data quality and generates quality reports"""
    
    def __init__(self):
        self.logger_name = 'data_validation'
        self.validation_results = {}
    
    def validate_schema(self, df: pd.DataFrame, required_columns: List[str]) -> Dict:
        """Validate that required columns are present"""
        missing_columns = set(required_columns) - set(df.columns)
        extra_columns = set(df.columns) - set(required_columns)
        
        return {
            'has_required_columns': len(missing_columns) == 0,
            'missing_columns': list(missing_columns),
            'extra_columns': list(extra_columns),
            'total_columns': len(df.columns)
        }
    
    def validate_missing_values(self, df: pd.DataFrame) -> Dict:
        """Check for missing values"""
        # Exclude non-relevant columns from missing value checks
        exclude_columns = ['event_type','extra_column']
        if exclude_columns:
            columns_to_check = [col for col in df.columns if col not in exclude_columns]
        else:
            columns_to_check = df.columns

        # Calculate missing value counts for the selected columns
        missing_counts = df[columns_to_check].isnull().sum()
        
        missing_percentages = (missing_counts / len(df) * 100).round(2)

        return {
            'total_missing': int(missing_counts.sum()),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict(),
            'rows_with_any_missing': int(df.isnull().any(axis=1).sum())
        }
    
    def validate_duplicates(self, df: pd.DataFrame, subset: List[str] = None) -> Dict:
        """Check for duplicate records"""
        if subset:
            duplicates = df.duplicated(subset=subset, keep=False)
        else:
            duplicates = df.duplicated(keep=False)
        
        return {
            'total_duplicates': int(duplicates.sum()),
            'duplicate_percentage': round(duplicates.sum() / len(df) * 100, 2),
            'unique_rows': int((~df.duplicated()).sum())
        }
    
    def validate_range(self, df: pd.DataFrame, column: str, 
                      min_val: float, max_val: float) -> Dict:
        """Validate that numeric column is within expected range"""
        if column not in df.columns:
            return {'error': f'Column {column} not found'}
        
        out_of_range = (df[column] < min_val) | (df[column] > max_val)
        
        return {
            'column': column,
            'expected_range': [min_val, max_val],
            'actual_min': float(df[column].min()),
            'actual_max': float(df[column].max()),
            'out_of_range_count': int(out_of_range.sum()),
            'out_of_range_percentage': round(out_of_range.sum() / len(df) * 100, 2)
        }
    
    def validate_data_types(self, df: pd.DataFrame, 
                           expected_types: Dict[str, str]) -> Dict:
        """Validate column data types"""
        type_mismatches = {}
        
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type not in actual_type:
                    type_mismatches[col] = {
                        'expected': expected_type,
                        'actual': actual_type
                    }
        
        return {
            'type_mismatches': type_mismatches,
            'all_types_correct': len(type_mismatches) == 0
        }
    
    def validate_interactions(self, df: pd.DataFrame) -> Dict:
        """Comprehensive validation for user interaction data"""
        with PipelineLogger(self.logger_name) as logger:
            logger.info("Validating user interaction data")
            
            if df.empty:
                return {
                    'timestamp': datetime.now().isoformat(),
                    'dataset_info': {'rows': 0, 'columns': 0, 'memory_usage_mb': 0},
                    'schema': {'has_required_columns': False, 'missing_columns': [], 'extra_columns': [], 'total_columns': 0},
                    'missing_values': {'total_missing': 0, 'columns_with_missing': {}, 'missing_percentages': {}, 'rows_with_any_missing': 0},
                    'duplicates': {'total_duplicates': 0, 'duplicate_percentage': 0, 'unique_rows': 0},
                    'rating_range': {'column': 'rating', 'expected_range': [1, 5], 'actual_min': None, 'actual_max': None, 'out_of_range_count': 0, 'out_of_range_percentage': 0},
                    'data_types': {'type_mismatches': {}, 'all_types_correct': True},
                    'statistics': {'unique_users': 0, 'unique_items': 0, 'avg_rating': None, 'rating_std': None},
                    'quality_score': 0
                }
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'dataset_info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
                }
            }
            
            required_cols = ['user_id', 'item_id', 'rating', 'timestamp']
            results['schema'] = self.validate_schema(df, required_cols)
            
            results['missing_values'] = self.validate_missing_values(df)
            
            results['duplicates'] = self.validate_duplicates(
                df, subset=['user_id', 'item_id']
            )
            
            if 'rating' in df.columns:
                results['rating_range'] = self.validate_range(df, 'rating', 1, 5)
            
            results['data_types'] = self.validate_data_types(df, {
                'user_id': 'int',
                'item_id': 'int',
                'rating': 'int'
            })
            
            results['statistics'] = {
                'unique_users': int(df['user_id'].nunique()) if 'user_id' in df.columns else 0,
                'unique_items': int(df['item_id'].nunique()) if 'item_id' in df.columns else 0,
                'avg_rating': float(df['rating'].mean()) if 'rating' in df.columns else 0,
                'rating_std': float(df['rating'].std()) if 'rating' in df.columns else 0
            }
            
            quality_score = self._calculate_quality_score(results)
            results['quality_score'] = quality_score
            
            logger.info(f"Validation completed. Quality score: {quality_score}/100")
            
            self.validation_results = results

            return results
    
    def _calculate_quality_score(self, results: Dict) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100.0
        
        # Avoid division by zero
        total_rows = results['dataset_info']['rows']
        if total_rows == 0:
            return 0.0
        
        if not results['schema']['has_required_columns']:
            score -= 30
        
        missing_pct = (results['missing_values']['total_missing'] / 
                      results['dataset_info']['rows'] * 100)
        score -= min(missing_pct, 20)
        
        dup_pct = results['duplicates']['duplicate_percentage']
        score -= min(dup_pct, 20)
        
        if 'rating_range' in results:
            out_of_range_pct = results['rating_range']['out_of_range_percentage']
            score -= min(out_of_range_pct, 15)
        
        if not results['data_types']['all_types_correct']:
            score -= 15
            
        # if 'extra_columns' in results['schema'] and results['schema']['extra_columns']:
        #     score -= 10  # Deduct points for extra columns
        
        return max(0, round(score, 2))
    
    def generate_report(self, output_path: str = 'reports/data_quality_report.json'):
        """Generate and save data quality report"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("DATA QUALITY REPORT")
        print('='*60)
        print(f"Quality Score: {self.validation_results.get('quality_score', 0)}/100")
        print(f"Total Rows: {self.validation_results['dataset_info']['rows']:,}")
        print(f"Total Columns: {self.validation_results['dataset_info']['columns']}")
        print(f"\nMissing Values: {self.validation_results['missing_values']['total_missing']:,}")
        print(f"Duplicate Records: {self.validation_results['duplicates']['total_duplicates']:,}")
        print(f"\nReport saved to: {output_path}")
        print('='*60)
        
    from fpdf import FPDF
    import json

    def generate_pdf_report(self, pdf_file: str):
        """
        Generate a PDF report directly from self.validation_results.
        """
        if not self.validation_results:
            print("No validation results available to generate the PDF report.")
            return

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        pdf.cell(200, 10, txt="Data Quality Report", ln=True, align='C')
        pdf.ln(10)
        
        for key, value in self.validation_results.items():
            if isinstance(value, dict):  # Handle nested dictionaries
                pdf.cell(200, 10, txt=f"{key}:", ln=True)
                for sub_key, sub_value in value.items():
                    pdf.cell(200, 10, txt=f"  {sub_key}: {sub_value}", ln=True)
            else:
                pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
        
        pdf.output(pdf_file)
        print(f"PDF report saved to {pdf_file}")

if __name__ == "__main__":
    df = pd.read_csv('data/raw/user_interactions.csv')
    
    validator = DataValidator()
    results = validator.validate_interactions(df)
    validator.generate_report()
    validator.generate_pdf_report(
        json_file="reports/data_quality_report.json",
        pdf_file="reports/data_quality_report.pdf"
    )