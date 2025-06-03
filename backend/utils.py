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
        if y_col:
            if not pd.api.types.is_numeric_dtype(df[y_col]): print(f"UTILS_RESHAPE_ERROR: y_column '{y_col}' not numeric."); return None
            pivot_df = df.pivot_table(index=x_col, columns=color_by_col, values=y_col, aggfunc=aggfunc).fillna(0)
        else:
            pivot_df = df.groupby([x_col, color_by_col]).size().unstack(fill_value=0)
        if pivot_df.empty: print("UTILS_RESHAPE_WARNING: Pivot table for stacked bar empty."); return None
        try:
            sorted_columns = pivot_df.sum().sort_values(ascending=False).index
            pivot_df = pivot_df[sorted_columns]
        except Exception as e_sort: print(f"UTILS_RESHAPE_WARNING: Could not sort columns for stacked bar: {e_sort}")
        print(f"UTILS_RESHAPE: Reshaped df for stacked bar. Head:\n{pivot_df.head()}")
        return pivot_df
    except Exception as e: print(f"UTILS_RESHAPE_CRITICAL_ERROR: {e}"); traceback.print_exc(); return None

def generate_plot_from_config(df: pd.DataFrame, config: PlotConfig) -> Optional[str]:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("UTILS_GENERATE_PLOT: NATIVE PLOTLY HIST/BOX, MPL FALLBACK (KDE line fix)") # Updated version
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"UTILS_GENERATE_PLOT: Received config: {config.model_dump_json(indent=2)}")
    print(f"UTILS_GENERATE_PLOT: df columns available: {df.columns.tolist()}")

    if df.empty: print("UTILS_GENERATE_PLOT_ERROR: DataFrame is empty."); return None

    actual_plot_type = config.plot_type
    if config.plot_type == "auto_categorical": actual_plot_type = "bar" # Default

    # --- NATIVE PLOTLY PATHS for BOXPLOT and HISTOGRAM ---
    if actual_plot_type == "boxplot":
        print("UTILS_NATIVE_PLOTLY: Generating BOXPLOT directly with Plotly Express.")
        try:
            if not config.y_column or config.y_column not in df.columns or not pd.api.types.is_numeric_dtype(df[config.y_column]):
                print(f"UTILS_NATIVE_PLOTLY_BOX_ERROR: Boxplot requires valid numeric y_column ('{config.y_column}')."); return None
            # Optional columns check
            if config.x_column and config.x_column not in df.columns: print(f"UTILS_NATIVE_PLOTLY_BOX_WARN: X-column '{config.x_column}' not found, proceeding without it."); config.x_column = None
            if config.color_by_column and config.color_by_column not in df.columns: print(f"UTILS_NATIVE_PLOTLY_BOX_WARN: Color-by '{config.color_by_column}' not found, proceeding without it."); config.color_by_column = None

            default_title = f"Boxplot of {config.y_column}{' by ' + config.x_column if config.x_column else ''}"
            plotly_fig_native = px.box(
                df, x=config.x_column, y=config.y_column, color=config.color_by_column,
                title=config.title or default_title,
                labels={config.y_column: config.ylabel or config.y_column,
                        config.x_column if config.x_column else "_": config.xlabel or config.x_column or ("" if not config.x_column else "Category")},
                points="outliers"
            )
            plotly_fig_native.update_layout(xaxis_title_text=config.xlabel or config.x_column or "", yaxis_title_text=config.ylabel or config.y_column, legend_title_text=config.color_by_column or "")
            print("UTILS_NATIVE_PLOTLY: Boxplot generated with Plotly Express.")
            return pio.to_json(plotly_fig_native)
        except Exception as e_native_px_box: print(f"UTILS_NATIVE_PLOTLY_BOX_ERROR: {e_native_px_box}"); traceback.print_exc(); return None

    elif actual_plot_type == "histogram":
        print("UTILS_NATIVE_PLOTLY: Generating HISTOGRAM directly with Plotly Express.")
        try:
            if not config.x_column or config.x_column not in df.columns or not pd.api.types.is_numeric_dtype(df[config.x_column]):
                print(f"UTILS_NATIVE_PLOTLY_HIST_ERROR: Histogram requires valid numeric x_column ('{config.x_column}')."); return None
            if config.color_by_column and config.color_by_column not in df.columns: print(f"UTILS_NATIVE_PLOTLY_HIST_WARN: Color-by '{config.color_by_column}' not found, proceeding without it."); config.color_by_column = None

            default_title = f"Histogram of {config.x_column}{' by ' + config.color_by_column if config.color_by_column else ''}"
            # Plotly express histogram can show a KDE curve using marginal="rug" (or "box", "violin")
            # and then one can add a KDE trace manually if needed, or accept hist + rug.
            # For a direct replacement of sns.histplot(kde=True), we'd often want density normalization.
            plotly_fig_native = px.histogram(
                df, x=config.x_column, color=config.color_by_column,
                marginal="rug", # Shows distribution of points, good with histograms
                # histnorm='probability density', # Use this if you want normalized heights for KDE-like appearance
                nbins=config.bins, # Pass user-specified bins if any
                title=config.title or default_title,
                labels={config.x_column: config.xlabel or config.x_column}
            )
            # If you want a KDE overlay similar to sns.histplot(kde=True)
            # This requires more work with Plotly: either using `fig.add_trace(go.Scatter(...))` with calculated KDE data
            # or using a library that helps create KDE traces for Plotly.
            # For simplicity now, px.histogram with "rug" gives some sense of distribution.
            # If histnorm='probability density' is used, the bars sum to 1 (area).
            plotly_fig_native.update_layout(
                xaxis_title_text=config.xlabel or config.x_column,
                yaxis_title_text=config.ylabel or "Frequency", # Plotly hist default y is count/frequency
                legend_title_text=config.color_by_column or "",
                bargap=0.1 if not config.color_by_column else 0.2 # Default bargap for px.histogram
            )
            # If `config.color_by_column` is used, px.histogram might default to `barmode='stack'` or `'overlay'`
            # If it's overlay and you want grouped, you'd set: plotly_fig_native.update_layout(barmode='group')

            print("UTILS_NATIVE_PLOTLY: Histogram generated with Plotly Express.")
            return pio.to_json(plotly_fig_native)
        except Exception as e_native_px_hist: print(f"UTILS_NATIVE_PLOTLY_HIST_ERROR: {e_native_px_hist}"); traceback.print_exc(); return None


    # --- MATPLOTLIB/SEABORN PATH FOR OTHER PLOTS (BAR, KDE, LINE, SCATTER) ---
    print(f"UTILS_MPL_PLOT: Matplotlib path for plot type: '{actual_plot_type}'")
    # (Sizing logic from previous version - ENSURE THIS IS COMPLETE AND CORRECT)
    fig_width = 10; fig_height = 6; xtick_rotation = 0; xtick_ha = 'center'; xtick_fontsize = 10
    final_rect_bottom_margin = 0.12; final_rect_right_margin = 0.95; num_x_categories = 0
    if config.x_column and config.x_column in df.columns: num_x_categories = df[config.x_column].nunique()
    elif actual_plot_type in ["bar", "kde", "line", "scatter"]: print(f"UTILS_MPL_ERR: X-col '{config.x_column}' required/not found for '{actual_plot_type}'."); return None
    if actual_plot_type == "bar":
        xtick_fontsize=9
        if num_x_categories > 8: fig_width,fig_height,xtick_rotation,xtick_ha,final_rect_bottom_margin=max(10,min(num_x_categories*0.7,35)),max(6.5,min(6+num_x_categories*0.1,12)),45,'right',0.20
        if num_x_categories > 15: fig_width,fig_height,xtick_fontsize,final_rect_bottom_margin=max(12,min(num_x_categories*0.8,40)),max(7,min(6+num_x_categories*0.15,15)),8,0.25
        if num_x_categories > 25: xtick_rotation,final_rect_bottom_margin=60,0.30
    elif actual_plot_type == "kde": # Sizing for KDE (usually numeric X)
        fig_width,fig_height,final_rect_bottom_margin,xtick_rotation=12,7,0.15,0
    elif actual_plot_type == "scatter":
        if config.x_column and (df[config.x_column].dtype=='object' or pd.api.types.is_string_dtype(df[config.x_column])):
            if num_x_categories > 8: fig_width,fig_height,xtick_rotation,xtick_ha,xtick_fontsize,final_rect_bottom_margin=max(10,min(num_x_categories*0.6,30)),max(6.5,min(6+num_x_categories*0.1,12)),45,'right',9,0.20
    elif actual_plot_type == "line":
        if config.x_column and (df[config.x_column].dtype=='object' or pd.api.types.is_string_dtype(df[config.x_column])):
            if num_x_categories > 10: fig_width,fig_height,xtick_rotation,xtick_ha,xtick_fontsize,final_rect_bottom_margin=max(10,min(num_x_categories*0.6,30)),7,45,'right',9,0.20

    print(f"UTILS_MPL_PLOT: Matplotlib: type='{actual_plot_type}', fig_w={fig_width}, fig_h={fig_height}, rot={xtick_rotation}, margin_b={final_rect_bottom_margin}")
    plt.style.use('seaborn-v0_8-whitegrid'); fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    try:
        if config.y_column and config.y_column not in df.columns: print(f"UTILS_MPL_YCOL_ERR"); plt.close(fig); return None
        if config.color_by_column and config.color_by_column not in df.columns: print(f"UTILS_MPL_HUECOL_ERR"); plt.close(fig); return None
        plot_generated = False
        # MPL plotting logic for BAR, KDE, LINE, SCATTER
        if actual_plot_type == "bar":
            print("UTILS_MPL_PLOT: BAR"); bar_style="grouped";num_hue_cats=0
            if config.color_by_column: num_hue_cats=df[config.color_by_column].nunique()
            if config.color_by_column and (num_hue_cats>=5 or (num_hue_cats>2 and num_x_categories>=8)): bar_style="stacked"
            if bar_style=="stacked" and config.color_by_column:
                y_param=config.y_column;
                if config.y_column and not pd.api.types.is_numeric_dtype(df[config.y_column]): y_param=None
                reshaped=_reshape_for_stacked_bar(df,config.x_column,y_param,config.color_by_column,np.sum)
                if reshaped is not None and not reshaped.empty: reshaped.plot(kind='bar',stacked=True,ax=ax,width=0.8); plot_generated=True
                else: bar_style="grouped"
            if bar_style=="grouped" or not plot_generated:
                if config.y_column:
                    if not pd.api.types.is_numeric_dtype(df[config.y_column]): sns.countplot(data=df,x=config.x_column,hue=config.color_by_column,ax=ax)
                    else: sns.barplot(data=df,x=config.x_column,y=config.y_column,hue=config.color_by_column,estimator=np.sum,ax=ax)
                else: sns.countplot(data=df,x=config.x_column,hue=config.color_by_column,ax=ax)
                plot_generated=True
        elif actual_plot_type == "kde": # Histogram is now native Plotly
            print("UTILS_MPL_PLOT: KDE");
            if not pd.api.types.is_numeric_dtype(df[config.x_column]): print(f"UTILS_MPL_KDE_ERR: X not numeric"); plt.close(fig);return None
            if df[config.x_column].dropna().nunique()<2: print(f"UTILS_MPL_KDE_WARN: Not enough unique data for KDE")
            sns.kdeplot(data=df, x=config.x_column, hue=config.color_by_column, fill=False, linewidth=2, ax=ax); plot_generated=True
        elif actual_plot_type == "line":
            print("UTILS_MPL_PLOT: LINE"); df_s=df.copy()
            if config.x_column in df_s.columns and (pd.api.types.is_datetime64_any_dtype(df_s[config.x_column])or pd.api.types.is_numeric_dtype(df_s[config.x_column])): df_s=df_s.sort_values(by=config.x_column)
            if not config.y_column:
                cnt_col='_g_cnt_mpl';
                if pd.api.types.is_numeric_dtype(df_s[config.x_column]): sns.lineplot(data=df_s,x=config.x_column,y=df_s.index,hue=config.color_by_column,marker='o',ax=ax);ax.set_ylabel(config.ylabel or "Index")
                else:
                    if config.color_by_column: cnt_data=df_s.groupby([config.x_column,config.color_by_column],observed=False).size().reset_index(name=cnt_col); sns.lineplot(data=cnt_data,x=config.x_column,y=cnt_col,hue=config.color_by_column,marker='o',ax=ax)
                    else: cnt_data=df_s.groupby(config.x_column,observed=False).size().reset_index(name=cnt_col); sns.lineplot(data=cnt_data,x=config.x_column,y=cnt_col,marker='o',ax=ax)
                    ax.set_ylabel(config.ylabel or "Count")
            else:
                if not pd.api.types.is_numeric_dtype(df_s[config.y_column]):print(f"UTILS_MPL_LINE_WARN: Y not numeric")
                sns.lineplot(data=df_s,x=config.x_column,y=config.y_column,hue=config.color_by_column,marker='o',ax=ax)
            plot_generated=True
        elif actual_plot_type == "scatter":
            print("UTILS_MPL_PLOT: SCATTER");
            if not config.y_column: print("UTILS_MPL_SCATTER_ERR: Y-col required");plt.close(fig);return None
            if not pd.api.types.is_numeric_dtype(df[config.x_column]): print(f"UTILS_MPL_SCATTER_WARN: X not numeric")
            if not pd.api.types.is_numeric_dtype(df[config.y_column]): print(f"UTILS_MPL_SCATTER_WARN: Y not numeric")
            sns.scatterplot(data=df,x=config.x_column,y=config.y_column,hue=config.color_by_column,ax=ax);plot_generated=True

        if not plot_generated: print(f"UTILS_MPL_ERR: Plot type '{actual_plot_type}' not generated."); plt.close(fig); return None

        # (Titles, Labels, Ticks for MPL path - ENSURE COMPLETE)
        title_str=config.title or f"{actual_plot_type.capitalize()} of {config.x_column or 'Y'}"; def_y_lab="Value"
        if actual_plot_type=='bar' and not config.y_column and config.x_column: title_str,def_y_lab=config.title or f"Count of {config.x_column}","Count"
        elif actual_plot_type=='kde': def_y_lab="Density"
        elif actual_plot_type=='line' and (not config.y_column or (hasattr(ax,'get_ylabel')and'_g_cnt_mpl' in ax.get_ylabel())):def_y_lab="Count"
        ax.set_title(title_str,fontsize=14,wrap=True,pad=20);ax.set_xlabel(config.xlabel or config.x_column or "",fontsize=12);ax.set_ylabel(config.ylabel or config.y_column or def_y_lab,fontsize=12)
        if xtick_rotation>0 and config.x_column:ax.tick_params(axis='x',labelsize=xtick_fontsize);plt.setp(ax.get_xticklabels(),rotation=xtick_rotation,ha=xtick_ha)
        elif config.x_column:ax.tick_params(axis='x',labelsize=xtick_fontsize)
        ax.tick_params(axis='y',labelsize=10)

        # (Legend for MPL path - ENSURE COMPLETE)
        leg=ax.get_legend();fr_margin_orig=final_rect_right_margin
        if config.color_by_column and leg:
            h,l=[],[]
            try:
                _h,_l=leg.get_legend_handles_labels()
                if _h and _l:h,l=_h,_l
                elif hasattr(leg,'legendHandles')and hasattr(leg,'texts'):h,l=leg.legendHandles,[t.get_text() for t in leg.texts]
            except Exception as e_l:print(f"MPL_LEG_WARN:{e_l}")
            n_items=len(h)
            if n_items>20:leg.remove();final_rect_right_margin=fr_margin_orig
            elif n_items>0:
                ncol=1;
                if n_items>6:ncol=2
                if n_items>12 and fig_width>15:ncol=3
                ax.legend(h,l,title=(config.color_by_column or "Leg"),ncol=ncol,fontsize=8,bbox_to_anchor=(1.02,1),loc='upper left',frameon=True)
                if ncol==1:final_rect_right_margin=0.82
                elif ncol==2:final_rect_right_margin=0.75
                else:final_rect_right_margin=0.68
            else:
                if leg:leg.remove();final_rect_right_margin=fr_margin_orig
        elif leg:leg.remove();final_rect_right_margin=fr_margin_orig
        
        # (Final Layout for MPL - ENSURE COMPLETE)
        final_r=[0.08,final_rect_bottom_margin,final_rect_right_margin,0.92]
        if final_r[0]>=final_r[2]:final_r[2]=final_r[0]+0.1
        if final_r[1]>=final_r[3]:final_r[3]=final_r[1]+0.1
        print(f"UTILS_MPL_PLOT: tight_layout rect: {final_r}")
        try:fig.tight_layout(rect=final_r)
        except Exception as e_lo:print(f"MPL_LAYOUT_WARN:{e_lo}, trying default.");_=fig.tight_layout() if hasattr(fig,'tight_layout')else None

        # Convert MPL fig to Plotly JSON
        print("UTILS_MPL_PLOT: Converting Matplotlib figure to Plotly figure.")
        plotly_fig_mpl = None
        try:
            plotly_fig_mpl = plotly.tools.mpl_to_plotly(fig)
            # Bargap fix for MPL converted figures (BAR and HISTOGRAM handled by native, this is for safety if BAR goes via MPL)
            if plotly_fig_mpl and 'layout' in plotly_fig_mpl and isinstance(plotly_fig_mpl['layout'], dict):
                if 'bargap' in plotly_fig_mpl['layout']:
                    cur_bg = plotly_fig_mpl['layout']['bargap']
                    print(f"UTILS_MPL_PLOT: Initial bargap (mpl_to_plotly): {cur_bg} (type: {type(cur_bg)})")
                    if isinstance(cur_bg, (np.float64, float, int)): # np.float32 removed for brevity
                        new_bg = float(cur_bg)
                        if not (0 <= new_bg <= 1):
                            orig_bg = new_bg
                            if new_bg < 0 and new_bg > -1e-9: new_bg = 0.0
                            elif not (0 <= new_bg <= 1): new_bg = 0.2
                            print(f"UTILS_MPL_PLOT: Sanitizing bargap from {orig_bg} to {new_bg}")
                            plotly_fig_mpl['layout']['bargap'] = new_bg
                        else: plotly_fig_mpl['layout']['bargap'] = float(new_bg); print(f"UTILS_MPL_PLOT: bargap {new_bg} valid, ensured float.")
                # This elif is mainly for 'bar' if it ever goes through MPL path
                elif actual_plot_type in ["bar", "auto_categorical"]: # Histogram is native
                    plotly_fig_mpl['layout']['bargap'] = 0.2
                    print(f"UTILS_MPL_PLOT: bargap not in layout for {actual_plot_type}, set default 0.2")

            json_out = pio.to_json(plotly_fig_mpl)
            print("UTILS_MPL_PLOT: Matplotlib figure converted to Plotly JSON successfully.")
            return json_out
        except ValueError as ve_mpl_plotly: print(f"UTILS_MPL_PLOT_CRIT_ERR (ValueError): {ve_mpl_plotly} for {actual_plot_type}"); traceback.print_exc(); return None
        except Exception as e_mpl_conv_final: print(f"UTILS_MPL_PLOT_CRIT_ERR (General): {e_mpl_conv_final} for {actual_plot_type}"); traceback.print_exc(); return None
        finally: plt.close(fig)
    except Exception as e_mpl_main_block: print(f"UTILS_MPL_PLOT_CRIT_ERR (Main Block): {e_mpl_main_block} for {actual_plot_type}"); traceback.print_exc(); plt.close(fig) if 'fig' in locals() else None; return None

# calculate_age_from_dob remains the same
def calculate_age_from_dob(df: pd.DataFrame, dob_column_name: str) -> Optional[pd.Series]:
    print(f"UTILS: Parsing DOB: '{dob_column_name}'")
    if dob_column_name not in df.columns: print(f"UTILS_DOB_ERR: Col not found."); return None
    try:
        print(f"UTILS: DOB Sample: {df[dob_column_name].head().tolist()}")
        dob_s = pd.to_datetime(df[dob_column_name], errors='coerce')
        if dob_s.isnull().all():
            dob_s_df = pd.to_datetime(df[dob_column_name], dayfirst=True, errors='coerce')
            if dob_s_df.notna().sum() > 0 and (dob_s.notna().sum()==0 or dob_s_df.notna().sum()>dob_s.notna().sum()): dob_s=dob_s_df; print("UTILS: Used dayfirst parse.")
        if dob_s.isnull().all(): print(f"UTILS_DOB_ERR: Cannot parse dates."); return None
        vc,tc=dob_s.notnull().sum(),len(dob_s);print(f"UTILS: Parsed {vc}/{tc} dates.");
        if vc==0: return None
        print(f"UTILS: Parsed DOB sample (non-null): {dob_s.dropna().head().tolist()}")
        today=pd.Timestamp(datetime.today()); age=today.year-dob_s.dt.year
        mask=(dob_s.dt.month>today.month)|((dob_s.dt.month==today.month)&(dob_s.dt.day>today.day))
        age=age-mask.astype(int).where(dob_s.notna(),pd.NA); age=age.astype('Int64')
        print(f"UTILS: Ages sample: {age.dropna().head().tolist()}, dtype: {age.dtype}")
        if age.notnull().any():
            stats=age.dropna().astype(float)
            if not stats.empty: print(f"UTILS: Age stats: min={stats.min()}, max={stats.max()}, mean={stats.mean():.1f}")
        return age
    except Exception as e: print(f"UTILS_DOB_EXC: {e}"); traceback.print_exc(); return None
# --- END OF FILE utils.py ---
