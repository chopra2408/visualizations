# # --- START OF FILE utils.py ---
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import io
# from typing import Optional
# from datetime import datetime
# from models import PlotConfig # Ensure PlotConfig is importable from your models.py
# import traceback
# import numpy as np

# def _reshape_for_stacked_bar(df: pd.DataFrame, x_col: str, y_col: Optional[str], color_by_col: str, aggfunc=np.sum) -> Optional[pd.DataFrame]:
#     """Helper to reshape data for stacked bar plotting using pandas."""
#     print(f"UTILS_RESHAPE: Reshaping for stacked bar. x_col='{x_col}', y_col='{y_col}', color_by_col='{color_by_col}'")
#     try:
#         if y_col: # We have a specific value column to aggregate
#             if not pd.api.types.is_numeric_dtype(df[y_col]):
#                 print(f"UTILS_RESHAPE_ERROR: y_column '{y_col}' is not numeric. Cannot aggregate for stacked bar values.")
#                 return None
#             pivot_df = df.pivot_table(
#                 index=x_col,
#                 columns=color_by_col,
#                 values=y_col,
#                 aggfunc=aggfunc # Use provided aggfunc
#             ).fillna(0)
#         else: # We are counting occurrences (equivalent to countplot)
#             pivot_df = df.groupby([x_col, color_by_col]).size().unstack(fill_value=0)
        
#         if pivot_df.empty:
#             print("UTILS_RESHAPE_WARNING: Pivot table for stacked bar is empty.")
#             return None
            
#         # Optional: Sort columns (hue categories) by their total sum for potentially better legend/stack order
#         # This can make plots where some categories are much larger appear more logically ordered.
#         try:
#             sorted_columns = pivot_df.sum().sort_values(ascending=False).index
#             pivot_df = pivot_df[sorted_columns]
#         except Exception as e_sort:
#             print(f"UTILS_RESHAPE_WARNING: Could not sort columns for stacked bar: {e_sort}")

#         print(f"UTILS_RESHAPE: Reshaped df for stacked bar. Head:\n{pivot_df.head()}")
#         return pivot_df
#     except Exception as e:
#         print(f"UTILS_RESHAPE_CRITICAL_ERROR: Failed to reshape data for stacked bar: {e}")
#         traceback.print_exc()
#         return None

# def generate_plot_from_config(df: pd.DataFrame, config: PlotConfig) -> Optional[bytes]:
#     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     print("UTILS_GENERATE_PLOT: RUNNING VERSION FROM [SMART_PLOT_CHOICE_V2_FULL]")
#     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     print(f"UTILS_GENERATE_PLOT: Received config: {config.model_dump_json(indent=2)}")
#     print(f"UTILS_GENERATE_PLOT: df columns available: {df.columns.tolist()}")

#     if df.empty:
#         print("UTILS_GENERATE_PLOT_ERROR: DataFrame is empty. Cannot generate plot.")
#         return None

#     # --- Determine actual plot type if "auto_categorical" is used ---
#     actual_plot_type = config.plot_type
#     if config.plot_type == "auto_categorical":
#         actual_plot_type = "bar" # Default 'auto_categorical' to 'bar'
#         print(f"UTILS_GENERATE_PLOT_INFO: 'auto_categorical' chosen, defaulting to '{actual_plot_type}'.")

#     # --- Dynamic Sizing and Layout Parameters ---
#     fig_width = 10; fig_height = 6; xtick_rotation = 0; xtick_ha = 'center'; xtick_fontsize = 10
#     final_rect_bottom_margin = 0.12; final_rect_right_margin = 0.95 # Default right margin (no legend outside)

#     num_x_categories = 0
#     if config.x_column and config.x_column in df.columns:
#         num_x_categories = df[config.x_column].nunique()
#     else: # If x_column is not specified or not in df, some plots can't be made
#         if actual_plot_type in ["bar", "histogram", "kde", "line", "scatter"]: # Boxplot can work with only y
#              print(f"UTILS_GENERATE_PLOT_ERROR: X-column '{config.x_column}' is required for '{actual_plot_type}' or not found.")
#              return None


#     if actual_plot_type == "bar":
#         xtick_fontsize = 9
#         if num_x_categories > 8:
#             fig_width = max(10, min(num_x_categories * 0.7, 35))
#             fig_height = max(6.5, min(6 + num_x_categories * 0.1, 12))
#             xtick_rotation = 45; xtick_ha = 'right'; final_rect_bottom_margin = 0.20
#         if num_x_categories > 15:
#             fig_width = max(12, min(num_x_categories * 0.8, 40))
#             fig_height = max(7, min(6 + num_x_categories * 0.15, 15))
#             xtick_fontsize = 8; final_rect_bottom_margin = 0.25
#         if num_x_categories > 25:
#             xtick_rotation = 60; final_rect_bottom_margin = 0.30

#     elif actual_plot_type in ["histogram", "kde"]:
#         if num_x_categories > 15: # Less relevant for hist/kde unless x is treated as categorical
#             fig_width = 12; fig_height = 7; xtick_rotation = 30
#             xtick_ha = 'right'; xtick_fontsize = 9; final_rect_bottom_margin = 0.18
    
#     elif actual_plot_type == "scatter":
#         if config.x_column and (df[config.x_column].dtype == 'object' or pd.api.types.is_string_dtype(df[config.x_column])):
#             if num_x_categories > 8:
#                 fig_width = max(10, min(num_x_categories * 0.6, 30)); fig_height = max(6.5, min(6 + num_x_categories * 0.1, 12))
#                 xtick_rotation = 45; xtick_ha = 'right'; xtick_fontsize = 9; final_rect_bottom_margin = 0.20
    
#     elif actual_plot_type == "line": # Similar sizing logic if X is categorical for line
#         if config.x_column and (df[config.x_column].dtype == 'object' or pd.api.types.is_string_dtype(df[config.x_column])):
#              if num_x_categories > 10:
#                 fig_width = max(10, min(num_x_categories * 0.6, 30)); fig_height = 7
#                 xtick_rotation = 45; xtick_ha = 'right'; xtick_fontsize = 9; final_rect_bottom_margin = 0.20


#     print(f"UTILS_GENERATE_PLOT: Calculated fig_width={fig_width}, fig_height={fig_height}, xtick_rotation={xtick_rotation}")
#     plt.style.use('seaborn-v0_8-whitegrid')
#     fig, ax = plt.subplots(figsize=(fig_width, fig_height))

#     try:
#         # Column existence checks (already partially handled by x_column check above)
#         if config.y_column and config.y_column not in df.columns:
#             print(f"UTILS_GENERATE_PLOT_ERROR: Y-column '{config.y_column}' not found."); plt.close(fig); return None
#         if config.color_by_column and config.color_by_column not in df.columns:
#             print(f"UTILS_GENERATE_PLOT_ERROR: Color-by-column '{config.color_by_column}' not found."); plt.close(fig); return None

#         plot_generated = False

#         if actual_plot_type == "bar":
#             print("UTILS_GENERATE_PLOT: Plot type is BAR.")
#             # x_column is already checked to exist if we reached here
            
#             bar_plot_style = "grouped" # Default
#             num_color_by_categories = 0
#             if config.color_by_column: # This implies color_by_column exists in df.columns
#                 num_color_by_categories = df[config.color_by_column].nunique()

#             if config.color_by_column: # Only consider stacked/grouped if there's a hue dimension
#                 # Heuristic for stacked vs grouped:
#                 # If hue categories are many, or x categories are many, stacked is often better.
#                 # Grouped becomes too wide or too many tiny bars.
#                 if num_color_by_categories >= 5 or (num_color_by_categories > 2 and num_x_categories >= 8) :
#                     bar_plot_style = "stacked"
#                     print(f"UTILS_GENERATE_PLOT_INFO: Choosing STACKED bar style. num_color_by_categories={num_color_by_categories}, num_x_categories={num_x_categories}.")
#                 else:
#                     print(f"UTILS_GENERATE_PLOT_INFO: Choosing GROUPED bar style. num_color_by_categories={num_color_by_categories}, num_x_categories={num_x_categories}.")
            
#             if bar_plot_style == "stacked" and config.color_by_column:
#                 agg_func_to_use = np.sum # Default for y_column
#                 if config.y_column and not pd.api.types.is_numeric_dtype(df[config.y_column]):
#                     print(f"UTILS_GENERATE_PLOT_WARNING: Y-column '{config.y_column}' is not numeric for stacked bar value. Will count occurrences if possible.")
#                     # If y_column is not numeric, _reshape_for_stacked_bar with y_col would fail.
#                     # We should effectively treat this as a count, so pass y_col=None to reshape.
#                     reshaped_df = _reshape_for_stacked_bar(df, config.x_column, None, config.color_by_column)
#                 elif not config.y_column: # Counting occurrences
#                      reshaped_df = _reshape_for_stacked_bar(df, config.x_column, None, config.color_by_column)
#                 else: # y_column is present and presumably numeric
#                      reshaped_df = _reshape_for_stacked_bar(df, config.x_column, config.y_column, config.color_by_column, aggfunc=agg_func_to_use)

#                 if reshaped_df is not None and not reshaped_df.empty:
#                     reshaped_df.plot(kind='bar', stacked=True, ax=ax, width=0.8)
#                     # Legend for pandas plot is handled later in common legend logic
#                     plot_generated = True
#                 else:
#                     print("UTILS_GENERATE_PLOT_ERROR: Failed to reshape data for stacked bar or reshaped_df is empty. Attempting grouped.")
#                     bar_plot_style = "grouped" # Fallback
            
#             if bar_plot_style == "grouped" or not plot_generated : # If grouped by choice, or stacking failed
#                 # This block will execute if style is grouped OR if stacking was chosen but failed (plot_generated still false)
#                 if not plot_generated: print("UTILS_GENERATE_PLOT_INFO: Proceeding with GROUPED bar chart.")

#                 if config.y_column:
#                     if not pd.api.types.is_numeric_dtype(df[config.y_column]):
#                         print(f"UTILS_GENERATE_PLOT_WARNING: Y-column '{config.y_column}' not numeric for barplot. Attempting countplot on X.")
#                         sns.countplot(data=df, x=config.x_column, hue=config.color_by_column, ax=ax)
#                     else:
#                         sns.barplot(data=df, x=config.x_column, y=config.y_column, hue=config.color_by_column, estimator=np.sum, ax=ax)
#                 else: # Count occurrences
#                     sns.countplot(data=df, x=config.x_column, hue=config.color_by_column, ax=ax)
#                 plot_generated = True
        
#         elif actual_plot_type == "histogram":
#             print("UTILS_GENERATE_PLOT: Plot type is HISTOGRAM.")
#             # x_column must exist (checked earlier)
#             if not pd.api.types.is_numeric_dtype(df[config.x_column]):
#                 print(f"UTILS_GENERATE_PLOT_ERROR: X-column '{config.x_column}' must be numeric for histogram."); plt.close(fig); return None
#             sns.histplot(data=df, x=config.x_column, hue=config.color_by_column, bins=config.bins or 'auto', kde=True, ax=ax)
#             plot_generated = True
        
#         elif actual_plot_type == "kde":
#             print("UTILS_GENERATE_PLOT: Plot type is KDE.")
#             if not pd.api.types.is_numeric_dtype(df[config.x_column]):
#                 print(f"UTILS_GENERATE_PLOT_ERROR: X-column '{config.x_column}' must be numeric for KDE plot."); plt.close(fig); return None
#             sns.kdeplot(data=df, x=config.x_column, hue=config.color_by_column, fill=True, ax=ax)
#             plot_generated = True

#         elif actual_plot_type == "scatter":
#             print("UTILS_GENERATE_PLOT: Plot type is SCATTER.")
#             if not config.y_column:
#                 print("UTILS_GENERATE_PLOT_ERROR: Scatter plot requires y_column."); plt.close(fig); return None
#             sns.scatterplot(data=df, x=config.x_column, y=config.y_column, hue=config.color_by_column, ax=ax)
#             plot_generated = True
        
#         elif actual_plot_type == "line":
#             print("UTILS_GENERATE_PLOT: Plot type is LINE.")
#             if not config.y_column:
#                 print("UTILS_GENERATE_PLOT_WARNING: Line plot ideally needs a y_column. Attempting to plot series or counts for x_column.")
#                 # This might need more sophisticated handling depending on desired behavior for 1-D line plots
#                 # For now, let's assume if y_column is missing, we might try to plot counts of x or just values if x is numeric
#                 if pd.api.types.is_numeric_dtype(df[config.x_column]):
#                     # Potentially sort by x if it's numeric for a coherent line
#                     df_sorted_line = df.sort_values(by=config.x_column) if config.x_column in df else df
#                     sns.lineplot(data=df_sorted_line, x=config.x_column, y=df_sorted_line.index, hue=config.color_by_column, marker='o', ax=ax) # Example y
#                     ax.set_ylabel(config.ylabel or "Index/Count", fontsize=10)
#                 else: # x is categorical, maybe plot counts?
#                     # This requires aggregation similar to countplot, then line plot
#                     print("UTILS_GENERATE_PLOT_ERROR: Line plot for categorical X without Y needs specific aggregation (e.g., counts over categories). Not implemented simply.")
#                     plt.close(fig); return None

#             else: # Both x and y columns are present
#                  # If x is datetime or numeric, sort by x for a meaningful line
#                 df_sorted_line = df
#                 if config.x_column in df.columns and (pd.api.types.is_datetime64_any_dtype(df[config.x_column]) or pd.api.types.is_numeric_dtype(df[config.x_column])):
#                     df_sorted_line = df.sort_values(by=config.x_column)

#                 sns.lineplot(data=df_sorted_line, x=config.x_column, y=config.y_column, hue=config.color_by_column, marker='o', ax=ax)
#             plot_generated = True
        
#         elif actual_plot_type == "boxplot":
#             print("UTILS_GENERATE_PLOT: Plot type is BOXPLOT.")
#             if not config.y_column or not pd.api.types.is_numeric_dtype(df[config.y_column]):
#                 print("UTILS_GENERATE_PLOT_ERROR: Boxplot requires a numeric y_column."); plt.close(fig); return None
            
#             if config.x_column: # Boxplot of y, grouped by x
#                 sns.boxplot(data=df, x=config.x_column, y=config.y_column, hue=config.color_by_column, ax=ax)
#             else: # Boxplot of y only (hue might be used if specified)
#                 sns.boxplot(data=df, y=config.y_column, hue=config.color_by_column, ax=ax)
#             plot_generated = True


#         if not plot_generated:
#             print(f"UTILS_GENERATE_PLOT_ERROR: Plot type '{actual_plot_type}' not handled successfully or prerequisites not met.")
#             plt.close(fig); return None

#         # --- Titles and Labels ---
#         title_str = config.title or f"{actual_plot_type.capitalize()} of {config.x_column or 'Y-axis'}"
#         if actual_plot_type == 'bar' and not config.y_column and config.x_column: # Count plot
#             title_str = config.title or f"Count of {config.x_column}"
#         ax.set_title(title_str, fontsize=14, wrap=True, pad=20)
#         ax.set_xlabel(config.xlabel or config.x_column or "", fontsize=12)

#         default_ylabel = "Value"
#         if actual_plot_type == 'bar' and not config.y_column: default_ylabel = "Count"
#         elif actual_plot_type == 'histogram': default_ylabel = "Frequency"
#         elif actual_plot_type == 'kde': default_ylabel = "Density"
#         ax.set_ylabel(config.ylabel or config.y_column or default_ylabel, fontsize=12)

#         if xtick_rotation > 0 and config.x_column:
#             ax.tick_params(axis='x', labelsize=xtick_fontsize)
#             plt.setp(ax.get_xticklabels(), rotation=xtick_rotation, ha=xtick_ha)
#         elif config.x_column :
#              ax.tick_params(axis='x', labelsize=xtick_fontsize)
        
#         ax.tick_params(axis='y', labelsize=10)

#         # --- CORRECTED Legend Handling ---
#         current_legend = ax.get_legend()
#         legend_present_and_outside = False # Initialize

#         if config.color_by_column and current_legend:
#             handles = []
#             labels = []

#             # Try to get handles and labels robustly
#             if hasattr(current_legend, 'legendHandles'): # Often for pandas legends
#                 handles = current_legend.legendHandles
#             elif hasattr(current_legend, 'legend_handles'): # Alternative attribute name
#                 handles = current_legend.legend_handles
#             elif hasattr(current_legend, '_legend_handle_box'): # For some matplotlib legends
#                  handles = [h for h,t in current_legend._legend_handle_box.get_children()]
#             elif hasattr(ax, 'legend_') and hasattr(ax.legend_, 'legendHandles'): # Another way pandas might store it
#                 handles = ax.legend_.legendHandles
            
#             if hasattr(current_legend, 'texts') and handles: # pandas often uses .texts for labels
#                 labels = [text.get_text() for text in current_legend.texts]
            
#             # Fallback if the above specific attributes for pandas aren't found,
#             # try the standard matplotlib way (which might fail if it's a pure pandas legend object)
#             if not handles or not labels:
#                 try:
#                     # This is the line that was causing the error for pandas legends
#                     # We only try it if other methods failed.
#                     _handles, _labels = current_legend.get_legend_handles_labels()
#                     if _handles and _labels: # Check if it returned valid lists
#                          handles = _handles
#                          labels = _labels
#                 except AttributeError:
#                     print("UTILS_GENERATE_PLOT_WARNING: Could not retrieve legend handles/labels using standard methods.")
#                     pass # handles/labels remain empty, legend might not be manageable

#             num_legend_items = len(handles)
#             print(f"UTILS_GENERATE_PLOT: Legend items count (after attempting retrieval): {num_legend_items}")

#             if num_legend_items > 20:
#                 print(f"UTILS_GENERATE_PLOT_WARNING: Too many legend items ({num_legend_items}), legend removed.")
#                 current_legend.remove()
#                 legend_present_and_outside = False
#             elif num_legend_items > 0:
#                 legend_ncol = 1
#                 if num_legend_items > 6: legend_ncol = 2
#                 if num_legend_items > 12 and fig_width > 15: legend_ncol = 3
                
#                 # Re-create the legend using obtained handles and labels for consistent control
#                 # This is safer than trying to modify the existing legend object in place,
#                 # especially when its type and properties vary.
#                 ax.legend(handles, labels, # Use the retrieved handles and labels
#                           title=(config.color_by_column or "Legend"),
#                           ncol=legend_ncol,
#                           fontsize=8,
#                           bbox_to_anchor=(1.02, 1),
#                           loc='upper left',
#                           frameon=True)
                
#                 legend_present_and_outside = True
#                 if legend_ncol == 1: final_rect_right_margin = 0.82
#                 elif legend_ncol == 2: final_rect_right_margin = 0.75
#                 else: final_rect_right_margin = 0.68
#                 print(f"UTILS_GENERATE_PLOT: Legend re-created with {legend_ncol} column(s). Right margin for plot area: {final_rect_right_margin}")
#             else: # No valid handles/labels found, remove inconsistent legend
#                 if current_legend: current_legend.remove()
#                 legend_present_and_outside = False
                
#         elif current_legend: # Legend exists but no color_by_column, remove it
#             current_legend.remove()
#             legend_present_and_outside = False
#         # else: no legend object at all, legend_present_and_outside remains False

#         # --- Final Layout Adjustment --- (Keep this part as is)
#         # ... (your existing tight_layout logic) ...
#         final_rect = [0.08, final_rect_bottom_margin, final_rect_right_margin, 0.92]
#         if final_rect[0] >= final_rect[2]: final_rect[2] = final_rect[0] + 0.1 
#         if final_rect[1] >= final_rect[3]: final_rect[3] = final_rect[1] + 0.1 

#         print(f"UTILS_GENERATE_PLOT: Applying tight_layout with rect: {final_rect}")
#         try:
#             fig.tight_layout(rect=final_rect)
#         except ValueError as ve_layout: 
#             print(f"UTILS_GENERATE_PLOT_WARNING: tight_layout with rect failed: {ve_layout}. Attempting default tight_layout.")
#             try: fig.tight_layout()
#             except Exception as e_layout_default:
#                  print(f"UTILS_GENERATE_PLOT_ERROR: Default tight_layout also failed: {e_layout_default}")
#         except Exception as e_layout_other: 
#             print(f"UTILS_GENERATE_PLOT_WARNING: tight_layout with rect failed with other error: {e_layout_other}. Attempting default tight_layout.")
#             try: fig.tight_layout()
#             except Exception as e_layout_default_other:
#                  print(f"UTILS_GENERATE_PLOT_ERROR: Default tight_layout also failed: {e_layout_default_other}")

#         buf = io.BytesIO()
#         print("UTILS_GENERATE_PLOT: Saving figure to buffer.")
#         plt.savefig(buf, format='png', bbox_inches='tight', dpi=120) # Increased DPI slightly for clarity
#         plt.close(fig)
#         buf.seek(0)
#         print("UTILS_GENERATE_PLOT: Figure saved, returning bytes.")
#         return buf.getvalue()

#     except Exception as e:
#         print(f"UTILS_GENERATE_PLOT_CRITICAL_ERROR: Main plotting block exception: {e}")
#         traceback.print_exc()
#         if 'fig' in locals() and fig is not None:
#             plt.close(fig)
#         return None

# def calculate_age_from_dob(df: pd.DataFrame, dob_column_name: str) -> Optional[pd.Series]:
#     """
#     Calculates age from a date of birth column, rounding to integers.
#     Returns a Pandas Series with ages as Int64 (nullable integer type), or None if calculation fails.
#     """
#     print(f"UTILS: Attempting to parse DOB column: '{dob_column_name}'")
#     if dob_column_name not in df.columns:
#         print(f"UTILS_ERROR: DOB column '{dob_column_name}' not found in DataFrame.")
#         return None

#     try:
#         print(f"UTILS: Sample DOB values (first 5, or fewer if less than 5 rows): {df[dob_column_name].head().tolist()}")

#         # Attempt parsing with multiple date formats
#         dob_series = pd.to_datetime(df[dob_column_name], dayfirst=True, errors='coerce')
#         if dob_series.isnull().all(): # if all are NaT, try default parsing
#             print("UTILS: First attempt (dayfirst=True) resulted in all NaT. Trying default parsing.")
#             dob_series = pd.to_datetime(df[dob_column_name], errors='coerce')
        
#         if dob_series.isnull().all():
#             print(f"UTILS_ERROR: DOB column '{dob_column_name}' could not be parsed into valid dates by any method.")
#             return None

#         valid_dates_count = dob_series.notnull().sum()
#         total_dates_count = len(dob_series)
#         print(f"UTILS: Successfully parsed {valid_dates_count}/{total_dates_count} dates from '{dob_column_name}'.")
#         if valid_dates_count == 0 :
#             print(f"UTILS_ERROR: No valid dates found in '{dob_column_name}' after parsing attempts.")
#             return None

#         print(f"UTILS: Parsed DOB sample (first 5 non-null): {dob_series.dropna().head().tolist()}")

#         today = pd.Timestamp(datetime.today())
#         age = today.year - dob_series.dt.year
        
#         mask_not_yet_bday = (dob_series.dt.month > today.month) | \
#                             ((dob_series.dt.month == today.month) & (dob_series.dt.day > today.day))
        
#         age = age - mask_not_yet_bday.astype(int) # NaT in dob_series or mask will propagate NaT to age

#         if age.notnull().any(): # If there's at least one non-NaT age
#              age = age.round(0).astype('Int64')
#         else: # all ages are NaT
#             print(f"UTILS: All calculated ages are NaT for column '{dob_column_name}'. Coercing to Int64 if possible.")
#             try:
#                 age = age.astype('Int64') # Attempt to make it Int64 even if all NaT
#             except TypeError: # This might happen if the series is object type from all NaTs
#                  print(f"UTILS_WARNING: Could not convert all-NaT age series to Int64. Returning as is (dtype: {age.dtype}).")

#         print(f"UTILS: Calculated ages sample (first 5 non-null): {age.dropna().head().tolist()}")
#         print(f"UTILS: Age dtype: {age.dtype}")
#         if age.notnull().any():
#             age_stats_series = age.dropna() # Calculate stats only on valid ages
#             if not age_stats_series.empty:
#                 print(f"UTILS: Age stats - min: {age_stats_series.min()}, max: {age_stats_series.max()}, mean: {age_stats_series.mean():.1f}")
#         return age

#     except Exception as e:
#         print(f"UTILS_ERROR: Error calculating age from '{dob_column_name}': {e}")
#         traceback.print_exc()
#         return None

# --- START OF FILE utils.py ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Optional
from datetime import datetime
from backend.models import PlotConfig
import traceback
import numpy as np
import plotly.tools
import plotly.io as pio # For converting to JSON
import plotly.express as px # For native Plotly plots

def _reshape_for_stacked_bar(df: pd.DataFrame, x_col: str, y_col: Optional[str], color_by_col: str, aggfunc=np.sum) -> Optional[pd.DataFrame]:
    """Helper to reshape data for stacked bar plotting using pandas."""
    print(f"UTILS_RESHAPE: Reshaping for stacked bar. x_col='{x_col}', y_col='{y_col}', color_by_col='{color_by_col}'")
    try:
        if y_col: # We have a specific value column to aggregate
            if not pd.api.types.is_numeric_dtype(df[y_col]):
                print(f"UTILS_RESHAPE_ERROR: y_column '{y_col}' is not numeric. Cannot aggregate for stacked bar values.")
                return None
            pivot_df = df.pivot_table(
                index=x_col,
                columns=color_by_col,
                values=y_col,
                aggfunc=aggfunc # Use provided aggfunc
            ).fillna(0)
        else: # We are counting occurrences (equivalent to countplot)
            pivot_df = df.groupby([x_col, color_by_col]).size().unstack(fill_value=0)

        if pivot_df.empty:
            print("UTILS_RESHAPE_WARNING: Pivot table for stacked bar is empty.")
            return None

        try:
            sorted_columns = pivot_df.sum().sort_values(ascending=False).index
            pivot_df = pivot_df[sorted_columns]
        except Exception as e_sort:
            print(f"UTILS_RESHAPE_WARNING: Could not sort columns for stacked bar: {e_sort}")

        print(f"UTILS_RESHAPE: Reshaped df for stacked bar. Head:\n{pivot_df.head()}")
        return pivot_df
    except Exception as e:
        print(f"UTILS_RESHAPE_CRITICAL_ERROR: Failed to reshape data for stacked bar: {e}")
        traceback.print_exc()
        return None

def generate_plot_from_config(df: pd.DataFrame, config: PlotConfig) -> Optional[str]:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("UTILS_GENERATE_PLOT: NATIVE PLOTLY BOXPLOT, MPL FALLBACK (bargap, KDE line fix) - TRULY FULL CODE")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"UTILS_GENERATE_PLOT: Received config: {config.model_dump_json(indent=2)}")
    print(f"UTILS_GENERATE_PLOT: df columns available: {df.columns.tolist()}")

    if df.empty:
        print("UTILS_GENERATE_PLOT_ERROR: DataFrame is empty. Cannot generate plot.")
        return None

    actual_plot_type = config.plot_type
    if config.plot_type == "auto_categorical":
        actual_plot_type = "bar"
        print(f"UTILS_GENERATE_PLOT_INFO: 'auto_categorical' chosen, defaulting to '{actual_plot_type}'.")

    # --- NATIVE PLOTLY PATH FOR BOXPLOT ---
    if actual_plot_type == "boxplot":
        print("UTILS_NATIVE_PLOTLY: Generating BOXPLOT directly with Plotly Express.")
        try:
            # Validation for boxplot columns
            if not config.y_column or config.y_column not in df.columns or not pd.api.types.is_numeric_dtype(df[config.y_column]):
                print(f"UTILS_NATIVE_PLOTLY_BOX_ERROR: Boxplot requires a valid numeric y_column ('{config.y_column}'). Found: {df[config.y_column].dtype if config.y_column in df else 'Not Found'}.")
                return None
            if config.x_column and config.x_column not in df.columns: # x_column is optional for boxplot
                print(f"UTILS_NATIVE_PLOTLY_BOX_ERROR: X-column '{config.x_column}' for boxplot not found.")
                return None
            if config.color_by_column and config.color_by_column not in df.columns:
                print(f"UTILS_NATIVE_PLOTLY_BOX_ERROR: Color-by-column '{config.color_by_column}' for boxplot not found.")
                return None

            # Generate title if not provided
            default_title = f"Boxplot of {config.y_column}"
            if config.x_column: default_title += f" by {config.x_column}"
            if config.color_by_column: default_title += f" (colored by {config.color_by_column})"

            plotly_fig_native = px.box(
                df,
                x=config.x_column,
                y=config.y_column,
                color=config.color_by_column,
                title=config.title or default_title,
                labels={ # Ensure labels are set even if config ones are None
                    config.y_column: config.ylabel or config.y_column,
                    config.x_column if config.x_column else "_": config.xlabel or config.x_column or ("" if not config.x_column else "Category"), # Use dummy key if x_col is None
                    config.color_by_column if config.color_by_column else "__": config.color_by_column if config.color_by_column else ""
                },
                points="outliers" # Common default to show outliers
            )
            # Further layout updates for clarity
            plotly_fig_native.update_layout(
                xaxis_title_text=config.xlabel or config.x_column or ("" if not config.x_column else "Category"), # Explicitly set axis titles
                yaxis_title_text=config.ylabel or config.y_column,
                legend_title_text=config.color_by_column if config.color_by_column else "" # Title for the legend
            )
            print("UTILS_NATIVE_PLOTLY: Boxplot generated successfully with Plotly Express.")
            return pio.to_json(plotly_fig_native)
        except Exception as e_native_px:
            print(f"UTILS_NATIVE_PLOTLY_BOX_CRITICAL_ERROR: Failed to generate boxplot with Plotly Express: {e_native_px}")
            traceback.print_exc()
            return None # Fail if native boxplot generation has an issue

    # --- MATPLOTLIB/SEABORN PATH FOR OTHER PLOTS ---
    print(f"UTILS_MPL_PLOT: Matplotlib path for plot type: '{actual_plot_type}'")
    fig_width = 10; fig_height = 6; xtick_rotation = 0; xtick_ha = 'center'; xtick_fontsize = 10
    final_rect_bottom_margin = 0.12; final_rect_right_margin = 0.95
    num_x_categories = 0

    if config.x_column and config.x_column in df.columns:
        num_x_categories = df[config.x_column].nunique()
    elif actual_plot_type in ["bar", "histogram", "kde", "line", "scatter"]: # These require x_column
        print(f"UTILS_MPL_PLOT_ERROR: X-column '{config.x_column}' is required for Matplotlib plot type '{actual_plot_type}' or not found in df columns: {df.columns.tolist()}.")
        return None

    # Dynamic Sizing Logic
    if actual_plot_type == "bar":
        xtick_fontsize = 9
        if num_x_categories > 8: fig_width, fig_height, xtick_rotation, xtick_ha, final_rect_bottom_margin = max(10, min(num_x_categories * 0.7, 35)), max(6.5, min(6 + num_x_categories * 0.1, 12)), 45, 'right', 0.20
        if num_x_categories > 15: fig_width, fig_height, xtick_fontsize, final_rect_bottom_margin = max(12, min(num_x_categories * 0.8, 40)), max(7, min(6 + num_x_categories * 0.15, 15)), 8, 0.25
        if num_x_categories > 25: xtick_rotation, final_rect_bottom_margin = 60, 0.30
    elif actual_plot_type in ["histogram", "kde"]:
        fig_width, fig_height, final_rect_bottom_margin = 12, 7, 0.15
        if config.x_column and (df[config.x_column].dtype == 'object' or pd.api.types.is_string_dtype(df[config.x_column])):
            if num_x_categories > 15: xtick_rotation, xtick_ha, xtick_fontsize, final_rect_bottom_margin = 30, 'right', 9, 0.18
        else: xtick_rotation = 0
    elif actual_plot_type == "scatter":
        if config.x_column and (df[config.x_column].dtype == 'object' or pd.api.types.is_string_dtype(df[config.x_column])):
            if num_x_categories > 8: fig_width,fig_height,xtick_rotation,xtick_ha,xtick_fontsize,final_rect_bottom_margin = max(10,min(num_x_categories*0.6,30)),max(6.5,min(6+num_x_categories*0.1,12)),45,'right',9,0.20
    elif actual_plot_type == "line":
        if config.x_column and (df[config.x_column].dtype == 'object' or pd.api.types.is_string_dtype(df[config.x_column])):
            if num_x_categories > 10: fig_width,fig_height,xtick_rotation,xtick_ha,xtick_fontsize,final_rect_bottom_margin = max(10,min(num_x_categories*0.6,30)),7,45,'right',9,0.20

    print(f"UTILS_MPL_PLOT: Matplotlib: type='{actual_plot_type}', fig_width={fig_width}, fig_height={fig_height}, xtick_rotation={xtick_rotation}, bottom_margin={final_rect_bottom_margin}")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    try:
        # Column existence checks for y_column and color_by_column for MPL path
        if config.y_column and config.y_column not in df.columns:
            print(f"UTILS_MPL_PLOT_ERROR: Y-column '{config.y_column}' not found for Matplotlib plot."); plt.close(fig); return None
        if config.color_by_column and config.color_by_column not in df.columns:
            print(f"UTILS_MPL_PLOT_ERROR: Color-by-column '{config.color_by_column}' not found for Matplotlib plot."); plt.close(fig); return None

        plot_generated = False

        if actual_plot_type == "bar":
            print("UTILS_MPL_PLOT: Plot type is BAR.")
            bar_plot_style = "grouped"; num_color_by_categories = 0
            if config.color_by_column: num_color_by_categories = df[config.color_by_column].nunique()
            if config.color_by_column:
                if num_color_by_categories >= 5 or (num_color_by_categories > 2 and num_x_categories >= 8): bar_plot_style = "stacked"
            if bar_plot_style == "stacked" and config.color_by_column:
                reshaped_df_param_y_col = config.y_column
                if config.y_column and not pd.api.types.is_numeric_dtype(df[config.y_column]): reshaped_df_param_y_col = None
                reshaped_df = _reshape_for_stacked_bar(df, config.x_column, reshaped_df_param_y_col, config.color_by_column, aggfunc=np.sum)
                if reshaped_df is not None and not reshaped_df.empty: reshaped_df.plot(kind='bar', stacked=True, ax=ax, width=0.8); plot_generated = True
                else: bar_plot_style = "grouped" # Fallback if reshape fails
            if bar_plot_style == "grouped" or not plot_generated:
                if config.y_column:
                    if not pd.api.types.is_numeric_dtype(df[config.y_column]): sns.countplot(data=df, x=config.x_column, hue=config.color_by_column, ax=ax)
                    else: sns.barplot(data=df, x=config.x_column, y=config.y_column, hue=config.color_by_column, estimator=np.sum, ax=ax)
                else: sns.countplot(data=df, x=config.x_column, hue=config.color_by_column, ax=ax)
                plot_generated = True
        elif actual_plot_type == "histogram":
            print("UTILS_MPL_PLOT: Plot type is HISTOGRAM.")
            if not pd.api.types.is_numeric_dtype(df[config.x_column]): print(f"UTILS_MPL_PLOT_ERROR: X-column for HISTOGRAM not numeric."); plt.close(fig); return None
            sns.histplot(data=df, x=config.x_column, hue=config.color_by_column, bins=config.bins or 'auto', kde=True, ax=ax)
            plot_generated = True
        elif actual_plot_type == "kde":
            print("UTILS_MPL_PLOT: Plot type is KDE.")
            if not pd.api.types.is_numeric_dtype(df[config.x_column]): print(f"UTILS_MPL_PLOT_ERROR: X-column for KDE not numeric."); plt.close(fig); return None
            kde_data_series = df[config.x_column]
            if kde_data_series.dropna().nunique() < 2: print(f"UTILS_MPL_PLOT_WARNING: Insufficient unique data for KDE in '{config.x_column}'.")
            sns.kdeplot(data=df, x=config.x_column, hue=config.color_by_column, fill=False, linewidth=2, ax=ax) # fill=False
            plot_generated = True
        elif actual_plot_type == "scatter":
            print("UTILS_MPL_PLOT: Plot type is SCATTER.")
            if not config.y_column: print("UTILS_MPL_PLOT_ERROR: Y-column required for SCATTER."); plt.close(fig); return None
            if not pd.api.types.is_numeric_dtype(df[config.x_column]): print(f"UTILS_MPL_PLOT_WARNING: X-column '{config.x_column}' for SCATTER is not numeric, plot may be unusual.");
            if not pd.api.types.is_numeric_dtype(df[config.y_column]): print(f"UTILS_MPL_PLOT_WARNING: Y-column '{config.y_column}' for SCATTER is not numeric, plot may be unusual.");
            sns.scatterplot(data=df, x=config.x_column, y=config.y_column, hue=config.color_by_column, ax=ax)
            plot_generated = True
        elif actual_plot_type == "line":
            print("UTILS_MPL_PLOT: Plot type is LINE.")
            df_sorted_line = df.copy() # Use a copy to sort
            if config.x_column in df_sorted_line.columns and (pd.api.types.is_datetime64_any_dtype(df_sorted_line[config.x_column]) or pd.api.types.is_numeric_dtype(df_sorted_line[config.x_column])):
                df_sorted_line = df_sorted_line.sort_values(by=config.x_column)
            if not config.y_column:
                print("UTILS_MPL_PLOT_WARNING: Line plot without Y-column. Plotting counts or index.")
                count_col_name = '_g_cnt_line_mpl'
                if pd.api.types.is_numeric_dtype(df_sorted_line[config.x_column]):
                    sns.lineplot(data=df_sorted_line, x=config.x_column, y=df_sorted_line.index, hue=config.color_by_column, marker='o', ax=ax)
                    ax.set_ylabel(config.ylabel or "Index")
                else: # Categorical X
                    if config.color_by_column:
                        count_data = df_sorted_line.groupby([config.x_column, config.color_by_column], observed=False).size().reset_index(name=count_col_name)
                        sns.lineplot(data=count_data, x=config.x_column, y=count_col_name, hue=config.color_by_column, marker='o', ax=ax)
                    else:
                        count_data = df_sorted_line.groupby(config.x_column, observed=False).size().reset_index(name=count_col_name)
                        sns.lineplot(data=count_data, x=config.x_column, y=count_col_name, marker='o', ax=ax)
                    ax.set_ylabel(config.ylabel or "Count")
            else: # Y-column is present
                if not pd.api.types.is_numeric_dtype(df_sorted_line[config.y_column]): print(f"UTILS_MPL_PLOT_WARNING: Y-column '{config.y_column}' for LINE is not numeric.");
                sns.lineplot(data=df_sorted_line, x=config.x_column, y=config.y_column, hue=config.color_by_column, marker='o', ax=ax)
            plot_generated = True
        # Boxplot is handled by native Plotly path, so no MPL boxplot here.

        if not plot_generated:
            print(f"UTILS_MPL_PLOT_ERROR: Plot type '{actual_plot_type}' could not be generated by Matplotlib path."); plt.close(fig); return None

        # Titles, Labels, Ticks for Matplotlib path
        title_str = config.title or f"{actual_plot_type.capitalize()} of {config.x_column or 'Y-axis'}"
        default_ylabel = "Value" # Default
        if actual_plot_type == 'bar' and not config.y_column and config.x_column: title_str, default_ylabel = config.title or f"Count of {config.x_column}", "Count"
        elif actual_plot_type == 'histogram': default_ylabel = "Frequency"
        elif actual_plot_type == 'kde': default_ylabel = "Density"
        elif actual_plot_type == 'line' and (not config.y_column or (hasattr(ax, 'get_ylabel') and '_g_cnt_line_mpl' in ax.get_ylabel())): default_ylabel = "Count"
        ax.set_title(title_str, fontsize=14, wrap=True, pad=20)
        ax.set_xlabel(config.xlabel or config.x_column or "", fontsize=12)
        ax.set_ylabel(config.ylabel or config.y_column or default_ylabel, fontsize=12)
        if xtick_rotation > 0 and config.x_column: ax.tick_params(axis='x', labelsize=xtick_fontsize); plt.setp(ax.get_xticklabels(), rotation=xtick_rotation, ha=xtick_ha)
        elif config.x_column: ax.tick_params(axis='x', labelsize=xtick_fontsize) # Apply fontsize even if no rotation
        ax.tick_params(axis='y', labelsize=10)

        # Legend Handling for Matplotlib path
        current_legend = ax.get_legend(); final_rect_right_margin_original = final_rect_right_margin
        if config.color_by_column and current_legend:
            handles, labels = [], []
            try:
                _h, _l = current_legend.get_legend_handles_labels()
                if _h and _l: handles, labels = _h, _l
                elif hasattr(current_legend, 'legendHandles') and hasattr(current_legend, 'texts'): handles, labels = current_legend.legendHandles, [t.get_text() for t in current_legend.texts]
            except Exception as e_leg: print(f"UTILS_MPL_LEGEND_WARN: {e_leg}")
            num_legend_items = len(handles)
            if num_legend_items > 20: current_legend.remove(); final_rect_right_margin = final_rect_right_margin_original
            elif num_legend_items > 0:
                legend_ncol = 1;
                if num_legend_items > 6: legend_ncol = 2
                if num_legend_items > 12 and fig_width > 15: legend_ncol = 3
                ax.legend(handles, labels, title=(config.color_by_column or "Legend"), ncol=legend_ncol, fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
                if legend_ncol == 1: final_rect_right_margin = 0.82
                elif legend_ncol == 2: final_rect_right_margin = 0.75
                else: final_rect_right_margin = 0.68
            else:
                if current_legend: current_legend.remove(); final_rect_right_margin = final_rect_right_margin_original
        elif current_legend: current_legend.remove(); final_rect_right_margin = final_rect_right_margin_original

        # Final Layout Adjustment for Matplotlib path
        final_rect = [0.08, final_rect_bottom_margin, final_rect_right_margin, 0.92]
        if final_rect[0] >= final_rect[2]: final_rect[2] = final_rect[0] + 0.1
        if final_rect[1] >= final_rect[3]: final_rect[3] = final_rect[1] + 0.1
        print(f"UTILS_MPL_PLOT: Applying tight_layout with rect: {final_rect}")
        try: fig.tight_layout(rect=final_rect)
        except Exception as e_layout: print(f"UTILS_MPL_TIGHT_LAYOUT_WARN: {e_layout}, trying default."); _ = fig.tight_layout() if hasattr(fig, 'tight_layout') else None

        # CONVERT MATPLOTLIB FIG TO PLOTLY JSON
        print("UTILS_MPL_PLOT: Converting Matplotlib figure to Plotly figure.")
        plotly_fig_mpl_conv = None
        try:
            plotly_fig_mpl_conv = plotly.tools.mpl_to_plotly(fig)

            if plotly_fig_mpl_conv and 'layout' in plotly_fig_mpl_conv and isinstance(plotly_fig_mpl_conv['layout'], dict):
                if 'bargap' in plotly_fig_mpl_conv['layout']:
                    current_bargap = plotly_fig_mpl_conv['layout']['bargap']
                    print(f"UTILS_MPL_PLOT: Initial bargap: {current_bargap} (type: {type(current_bargap)})")
                    if isinstance(current_bargap, (np.float64, np.float32, float, int)):
                        new_bargap = float(current_bargap)
                        if not (0 <= new_bargap <= 1):
                            original_val_for_log = new_bargap
                            if new_bargap < 0 and new_bargap > -1e-9: new_bargap = 0.0
                            elif not (0 <= new_bargap <= 1): new_bargap = 0.2
                            print(f"UTILS_MPL_PLOT: Sanitizing bargap from {original_val_for_log} to {new_bargap}")
                            plotly_fig_mpl_conv['layout']['bargap'] = new_bargap
                        else:
                            plotly_fig_mpl_conv['layout']['bargap'] = float(new_bargap)
                            print(f"UTILS_MPL_PLOT: bargap {new_bargap} is valid, ensured Python float.")
                elif actual_plot_type in ["histogram", "bar", "auto_categorical"]:
                    plotly_fig_mpl_conv['layout']['bargap'] = 0.2
                    print(f"UTILS_MPL_PLOT: bargap not in layout, set default 0.2 for {actual_plot_type}")

            json_output = pio.to_json(plotly_fig_mpl_conv)
            print("UTILS_MPL_PLOT: Matplotlib figure converted to Plotly JSON successfully.")
            return json_output
        except ValueError as ve_plotly_mpl:
            print(f"UTILS_MPL_PLOT_CRITICAL_ERROR: ValueError for {actual_plot_type} (mpl_to_plotly/to_json): {ve_plotly_mpl}")
            traceback.print_exc()
            if plotly_fig_mpl_conv and 'layout' in plotly_fig_mpl_conv: print(f"Problematic layout (sample): {json.dumps(plotly_fig_mpl_conv['layout'], default=str)[:500]}")
            return None
        except Exception as e_mpl_conv:
            print(f"UTILS_MPL_PLOT_CRITICAL_ERROR: General error for {actual_plot_type} (mpl_to_plotly/to_json): {e_mpl_conv}")
            traceback.print_exc()
            return None
        finally:
            plt.close(fig)

    except Exception as e_mpl_main:
        print(f"UTILS_MPL_PLOT_CRITICAL_ERROR: Main Matplotlib block exception for {actual_plot_type}: {e_mpl_main}")
        traceback.print_exc()
        if 'fig' in locals() and fig is not None: plt.close(fig)
        return None

def calculate_age_from_dob(df: pd.DataFrame, dob_column_name: str) -> Optional[pd.Series]:
    print(f"UTILS: Attempting to parse DOB column: '{dob_column_name}'")
    if dob_column_name not in df.columns:
        print(f"UTILS_ERROR: DOB column '{dob_column_name}' not found in DataFrame.")
        return None
    try:
        print(f"UTILS: Sample DOB values (first 5): {df[dob_column_name].head().tolist()}")
        dob_series = pd.to_datetime(df[dob_column_name], errors='coerce')
        if dob_series.isnull().all():
            print("UTILS: Default parsing resulted in all NaT. Trying with dayfirst=True.")
            dob_series_dayfirst = pd.to_datetime(df[dob_column_name], dayfirst=True, errors='coerce')
            if dob_series_dayfirst.notna().sum() > 0 and (dob_series.notna().sum() == 0 or dob_series_dayfirst.notna().sum() > dob_series.notna().sum()):
                dob_series = dob_series_dayfirst
                print("UTILS: Used dayfirst=True parsing results.")

        if dob_series.isnull().all():
            print(f"UTILS_ERROR: DOB column '{dob_column_name}' could not be parsed by any method.")
            return None

        valid_c, total_c = dob_series.notnull().sum(), len(dob_series)
        print(f"UTILS: Successfully parsed {valid_c}/{total_c} dates from '{dob_column_name}'.")
        if valid_c == 0: return None

        print(f"UTILS: Parsed DOB sample (first 5 non-null): {dob_series.dropna().head().tolist()}")
        today = pd.Timestamp(datetime.today())
        age = today.year - dob_series.dt.year
        mask_not_yet_bday = (dob_series.dt.month > today.month) | \
                            ((dob_series.dt.month == today.month) & (dob_series.dt.day > today.day))
        age = age - mask_not_yet_bday.astype(int).where(dob_series.notna(), pd.NA)
        age = age.astype('Int64') # Convert to nullable integer type

        print(f"UTILS: Calculated ages sample (first 5 non-null): {age.dropna().head().tolist()}")
        print(f"UTILS: Age dtype: {age.dtype}")
        if age.notnull().any():
            age_stats_series = age.dropna().astype(float)
            if not age_stats_series.empty:
                print(f"UTILS: Age stats - min: {age_stats_series.min()}, max: {age_stats_series.max()}, mean: {age_stats_series.mean():.1f}")
        return age
    except Exception as e:
        print(f"UTILS_ERROR: Error calculating age from '{dob_column_name}': {e}")
        traceback.print_exc()
        return None
# --- END OF FILE utils.py ---