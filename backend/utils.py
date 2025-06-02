import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from typing import Optional
from datetime import datetime
from backend.models import PlotConfig 
import traceback

def generate_plot_from_config(df: pd.DataFrame, config: PlotConfig) -> Optional[bytes]:
    print(f"UTILS_GENERATE_PLOT: Received config: {config.model_dump_json(indent=2)}")
    print(f"UTILS_GENERATE_PLOT: df columns available: {df.columns.tolist()}")

    if df.empty:
        print("UTILS_GENERATE_PLOT_ERROR: DataFrame is empty. Cannot generate plot.")
        return None

    fig_width = 12
    fig_height = 7

    if config.x_column and config.x_column in df.columns and df[config.x_column].nunique() > 10:
        num_categories = df[config.x_column].nunique()
        fig_width = max(12, min(num_categories * 0.7, 30))
        fig_height = 8
        print(f"UTILS_GENERATE_PLOT: Dynamic figsize used: ({fig_width}, {fig_height}) for {num_categories} categories.")
    
    plt.figure(figsize=(fig_width, fig_height)) 
    ax = None

    try:
        if config.x_column and config.x_column not in df.columns:
            print(f"UTILS_GENERATE_PLOT_ERROR: X-column '{config.x_column}' not found.")
            return None
        if config.y_column and config.y_column not in df.columns:
            print(f"UTILS_GENERATE_PLOT_ERROR: Y-column '{config.y_column}' not found.")
            return None
        if config.color_by_column and config.color_by_column not in df.columns:
            print(f"UTILS_GENERATE_PLOT_ERROR: Color-by-column '{config.color_by_column}' not found.")
            return None

        plot_generated = False

        if config.plot_type == "bar":
            print("UTILS_GENERATE_PLOT: Plot type is BAR.")
            if config.x_column:
                if config.y_column: 
                    if not pd.api.types.is_numeric_dtype(df[config.y_column]):
                        print(f"UTILS_GENERATE_PLOT_ERROR: Y-column '{config.y_column}' not numeric for barplot. Attempting countplot.")
                        ax = sns.countplot(data=df, x=config.x_column, hue=config.color_by_column)
                    else:
                        ax = sns.barplot(data=df, x=config.x_column, y=config.y_column, hue=config.color_by_column, estimator=sum)
                else:  
                    ax = sns.countplot(data=df, x=config.x_column, hue=config.color_by_column)
                plot_generated = True
                plt.xticks(rotation=60, ha='right', fontsize=9)
            else:
                print("UTILS_GENERATE_PLOT_ERROR: Bar plot requires an x_column.")
        
        elif config.plot_type == "histogram":
            print("UTILS_GENERATE_PLOT: Plot type is HISTOGRAM.")
            if config.x_column:
                if not pd.api.types.is_numeric_dtype(df[config.x_column]):
                    print(f"UTILS_GENERATE_PLOT_ERROR: X-column '{config.x_column}' must be numeric for histogram.")
                    return None
                ax = sns.histplot(data=df, x=config.x_column, hue=config.color_by_column, bins=config.bins or 'auto', kde=True)
                plot_generated = True
                plt.xticks(rotation=45, ha='right', fontsize=10)
            else:
                print("UTILS_GENERATE_PLOT_ERROR: Histogram requires an x_column.")

        elif config.plot_type == "kde":
            print("UTILS_GENERATE_PLOT: Plot type is KDE.")
            if config.x_column:
                if not pd.api.types.is_numeric_dtype(df[config.x_column]):
                    print(f"UTILS_GENERATE_PLOT_ERROR: X-column '{config.x_column}' must be numeric for KDE plot.")
                    return None
                ax = sns.kdeplot(data=df, x=config.x_column, hue=config.color_by_column, fill=True)
                plot_generated = True
                plt.xticks(rotation=45, ha='right', fontsize=10)
            else:
                print("UTILS_GENERATE_PLOT_ERROR: KDE plot requires an x_column.")

        elif config.plot_type == "scatter":
            print("UTILS_GENERATE_PLOT: Plot type is SCATTER.")
            if config.x_column and config.y_column:
                ax = sns.scatterplot(data=df, x=config.x_column, y=config.y_column, hue=config.color_by_column)
                plot_generated = True
                if df[config.x_column].dtype == 'object' or pd.api.types.is_string_dtype(df[config.x_column]):
                     print(f"UTILS_GENERATE_PLOT: Applying xticks rotation for scatter plot on column '{config.x_column}'.")
                     plt.xticks(rotation=45, ha='right', fontsize=10)
            else:
                print("UTILS_GENERATE_PLOT_ERROR: Scatter plot requires x_column and y_column.")

        if not plot_generated:
            print(f"UTILS_GENERATE_PLOT_ERROR: Plot type '{config.plot_type}' not handled or prereqs not met.")
            plt.close()
            return None

        title_str = config.title or f"{config.plot_type.capitalize()} of {config.x_column or ''}"
        plt.title(title_str, fontsize=12, wrap=True)
        plt.xlabel(config.xlabel or config.x_column, fontsize=10)
        plt.ylabel(config.ylabel or config.y_column or ("Count" if config.plot_type=='bar' and not config.y_column else "Value"), fontsize=10)

        if ax and config.color_by_column:
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                print(f"UTILS_GENERATE_PLOT: Placing legend. Number of items: {len(handles)}")
                if len(handles) > 15:
                    print("UTILS_GENERATE_PLOT_WARNING: Many legend items (>15), legend might be cluttered or removed.")
                    if ax.get_legend() is not None: ax.get_legend().remove()
                else:
                    ax.legend(handles, labels, title=config.color_by_column, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
            elif ax.get_legend() is not None:
                ax.get_legend().remove()
        elif ax and ax.get_legend() is not None:
             ax.get_legend().remove()

        print("UTILS_GENERATE_PLOT: Applying tight_layout.")
        try:
            if ax and config.color_by_column and handles and labels and len(handles) <=15 :
                 plt.tight_layout(rect=[0, 0, 0.85, 0.95])
            else:
                 plt.tight_layout(rect=[0, 0, 1, 0.95])
        except Exception as e_layout:
            print(f"UTILS_GENERATE_PLOT_WARNING: tight_layout failed: {e_layout}")

        buf = io.BytesIO()
        print("UTILS_GENERATE_PLOT: Saving figure to buffer.")
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        print("UTILS_GENERATE_PLOT: Figure saved, returning bytes.")
        return buf.getvalue()

    except Exception as e:
        print(f"UTILS_GENERATE_PLOT_CRITICAL_ERROR: Exception: {e}")
        traceback.print_exc()
        plt.close()
        return None

def calculate_age_from_dob(df: pd.DataFrame, dob_column_name: str) -> Optional[pd.Series]:
    """
    Calculates age from a date of birth column, rounding to integers.
    Returns a Pandas Series with ages as Int64 (nullable integer type), or None if calculation fails.
    """
    print(f"UTILS: Attempting to parse DOB column: '{dob_column_name}'")
    if dob_column_name not in df.columns:
        print(f"UTILS_ERROR: DOB column '{dob_column_name}' not found in DataFrame.")
        return None

    try:
        print(f"UTILS: Sample DOB values: {df[dob_column_name].head().tolist()}")

        # Attempt parsing with multiple date formats
        dob_series = pd.to_datetime(df[dob_column_name], dayfirst=True, errors='coerce')
        if dob_series.isnull().all():
            dob_series = pd.to_datetime(df[dob_column_name], errors='coerce')
        if dob_series.isnull().all():
            print(f"UTILS_ERROR: DOB column '{dob_column_name}' could not be parsed into valid dates.")
            return None

        print(f"UTILS: Parsed DOB sample: {dob_series.head().tolist()}")

        today = pd.Timestamp(datetime.today())
        age = today.year - dob_series.dt.year
        mask_not_yet_bday = (dob_series.dt.month > today.month) | \
                            ((dob_series.dt.month == today.month) & (dob_series.dt.day > today.day))
        age = age - mask_not_yet_bday.astype(int)

        # Round and convert to nullable integer type
        age = age.round(0).astype('Int64')
        print(f"UTILS: Calculated ages sample: {age.head().tolist()}")
        print(f"UTILS: Age dtype: {age.dtype}")
        print(f"UTILS: Age stats - min: {age.min()}, max: {age.max()}, mean: {age.mean():.1f}")
        return age

    except Exception as e:
        print(f"UTILS_ERROR: Error calculating age from '{dob_column_name}': {e}")
        traceback.print_exc()
        return None