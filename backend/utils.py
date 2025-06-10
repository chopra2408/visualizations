import pandas as pd
from typing import Optional, Tuple
from datetime import datetime
from backend.models import PlotConfig
import traceback
import numpy as np
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

def _apply_category_limits(
    df: pd.DataFrame,
    config: PlotConfig,
    target_col: str,
    value_col: Optional[str]  
) -> Tuple[pd.DataFrame, bool, Optional[str]]:
    """
    Applies top_n_categories or limit_categories_auto to the DataFrame.
    Returns the modified DataFrame, a boolean indicating if limiting was applied,
    and a string suffix for the plot title.
    """
    if not target_col or target_col not in df.columns:
        return df, False, None

    limited_df = df.copy()
    was_limited = False
    title_suffix = None
    
    num_unique_original = limited_df[target_col].nunique() # Use original df for this check
    limit_n = None

    # Determine the N for limiting
    if config.top_n_categories and config.top_n_categories > 0:
        limit_n = config.top_n_categories
        print(f"UTILS_LIMIT: Applying explicit top_n_categories: {limit_n} to '{target_col}'.")
    elif config.limit_categories_auto:
        default_auto_limit = 7 # General default, good for colors/facets
        if config.plot_type == "bar" and target_col == config.x_column:
            default_auto_limit = 12 # Allow more bars on x-axis
        elif config.plot_type == "pie" or config.plot_type == "doughnut":
            default_auto_limit = 6
        
        if num_unique_original > default_auto_limit:
            limit_n = default_auto_limit
            print(f"UTILS_LIMIT: Applying limit_categories_auto: {limit_n} to '{target_col}' (was {num_unique_original} unique).")
        else: # No need to limit if already below threshold
            return df, False, None

    if limit_n and num_unique_original > limit_n:
        was_limited = True
        metric_to_use = config.top_n_metric if config.top_n_categories else "sum"

        # Group by the target column and aggregate the value_col (or count)
        if value_col and value_col in limited_df.columns and pd.api.types.is_numeric_dtype(limited_df[value_col]):
            agg_func_map = {"sum": "sum", "mean": "mean", "median": "median", "count": "size"}
            agg_op = agg_func_map.get(metric_to_use, "sum") 

            if agg_op == "size":
                 grouped = limited_df.groupby(target_col, observed=True).size()
            else:
                 grouped = limited_df.groupby(target_col, observed=True)[value_col].agg(agg_op)
            
            top_groups = grouped.nlargest(limit_n).index
        else: 
            print(f"UTILS_LIMIT: Using frequency count for top-N on '{target_col}' (value_col:'{value_col}' invalid or metric='count').")
            top_groups = limited_df[target_col].value_counts().nlargest(limit_n).index
        
        limited_df = limited_df[limited_df[target_col].isin(top_groups)]
        # Make sure the limited_df[target_col] is categorical if it became non-categorical after filtering
        if not pd.api.types.is_categorical_dtype(limited_df[target_col]) and pd.api.types.is_object_dtype(limited_df[target_col]):
            limited_df[target_col] = pd.Categorical(limited_df[target_col], categories=top_groups, ordered=True)

        title_suffix = f" (Top {len(top_groups)} {target_col}s)"
    
    return limited_df, was_limited, title_suffix

def calculate_dynamic_height(base_height=450, item_height=250, num_items=0, wrap_cols=0, is_faceted=False):
    if not is_faceted or num_items <= 0 : # If not faceted, or no items to facet, use base_height
        return base_height
    if wrap_cols == 0 and num_items > 0 : # if faceting but no wrap, it's a single column/row of facets
        return max(base_height, item_height * num_items) # Stacked height
    if wrap_cols > 0 and num_items > 0:
        num_rows = (num_items + wrap_cols - 1) // wrap_cols
        return max(base_height, item_height * num_rows)
    return base_height # Default fallback

def generate_plot_from_config(df: pd.DataFrame, config: PlotConfig) -> Optional[str]:
    print(f"UTILS_GENERATE_PLOT: Received config: {config.model_dump_json(indent=0)}")

    if df.empty:
        print("UTILS_GENERATE_PLOT_ERROR: DataFrame is empty."); return None

    actual_plot_type = config.plot_type
    if config.plot_type == "auto_categorical":
        actual_plot_type = "bar"

    plot_df = df.copy() # Work with a copy to avoid modifying original session data
    final_title_base = config.title or ""
    additional_title_suffix = ""
    fig = None # Initialize fig to None

    # --- Plotly Express Plot Generation ---
    try:
        # Dynamic height calculation helper
        def calculate_dynamic_height(base_height=450, item_height=200, num_items=0, wrap_cols=0):
            if num_items <= 1 or wrap_cols == 0:
                return base_height
            num_rows = (num_items + wrap_cols - 1) // wrap_cols
            return max(base_height, item_height * num_rows)

        # Facet column wrap calculation helper
        def get_facet_wrap_and_count(df_for_facet: pd.DataFrame, facet_col_name: Optional[str]):
            wrap_val = 0
            num_facets = 0
            if facet_col_name and facet_col_name in df_for_facet.columns:
                num_facets = df_for_facet[facet_col_name].nunique()
                if num_facets > 2 and num_facets <= 12: # Sensible range for wrapping
                    wrap_val = 2 if num_facets <= 4 else 3 if num_facets <= 9 else 4
                elif num_facets > 12: # For very many facets, cap wrap to avoid tiny plots
                    wrap_val = 4
            return wrap_val, num_facets

        if actual_plot_type == "line":
            if not config.x_column or config.x_column not in plot_df.columns:
                print(f"UTILS_PX_LINE_ERROR: Line chart needs valid x_column ('{config.x_column}')."); return None
            
            y_col_for_plot = config.y_column
            current_color_col = config.color_by_column
            y_axis_label = config.ylabel or y_col_for_plot or "Value"

            if not final_title_base:
                final_title_base = f"Line Plot of {y_col_for_plot or 'Value'} over {config.x_column}"

            if current_color_col and config.aggregate_by_color_col_method and y_col_for_plot:
                print(f"UTILS_PX_LINE: Aggregating '{y_col_for_plot}' by '{config.aggregate_by_color_col_method}'.")
                if pd.api.types.is_datetime64_any_dtype(plot_df[config.x_column]):
                    plot_df = plot_df.sort_values(by=config.x_column)
                agg_op = config.aggregate_by_color_col_method
                aggregated_df = plot_df.groupby(config.x_column, as_index=False, observed=True).agg(
                    aggregated_y=(y_col_for_plot, agg_op)
                )
                plot_df = aggregated_df
                y_col_for_plot = 'aggregated_y'
                current_color_col = None # Aggregated, so no longer coloring by original
                y_axis_label = f"{agg_op.capitalize()} of {config.y_column}"
                additional_title_suffix += f" ({agg_op.capitalize()} across all {config.color_by_column}s)"
            elif current_color_col and current_color_col in plot_df.columns:
                plot_df, limited, suffix = _apply_category_limits(plot_df, config, current_color_col, y_col_for_plot)
                if limited and suffix: additional_title_suffix += suffix

            if pd.api.types.is_datetime64_any_dtype(plot_df[config.x_column]) or \
               pd.api.types.is_numeric_dtype(plot_df[config.x_column]):
                plot_df = plot_df.sort_values(by=config.x_column)
            
            facet_wrap, num_facets = get_facet_wrap_and_count(plot_df, config.facet_column)
            dynamic_height = calculate_dynamic_height(num_items=num_facets, wrap_cols=facet_wrap)

            fig = px.line(
                plot_df, x=config.x_column, y=y_col_for_plot, color=current_color_col,
                markers=True if plot_df[config.x_column].nunique() < 30 else False,
                facet_col=config.facet_column, facet_row=config.facet_row, facet_col_wrap=facet_wrap,
                height=dynamic_height
            )

        elif actual_plot_type == "bar":
            if not config.x_column or config.x_column not in plot_df.columns:
                print(f"UTILS_PX_BAR_ERROR: Bar chart needs valid x_column ('{config.x_column}')."); return None

            current_x_col = config.x_column
            y_col_for_plot = config.y_column
            current_color_col = config.color_by_column if config.color_by_column and config.color_by_column in plot_df.columns else None
            y_axis_label = config.ylabel
            is_count_plot = False
            temp_count_col_name = '_count_'

            if not final_title_base:
                final_title_base = f"Bar Chart of {current_x_col}"
                if current_color_col: final_title_base += f" by {current_color_col}"

            if not y_col_for_plot or (y_col_for_plot in plot_df.columns and not pd.api.types.is_numeric_dtype(plot_df[y_col_for_plot])):
                if y_col_for_plot: print(f"UTILS_PX_BAR_WARN: Y-column '{y_col_for_plot}' non-numeric. Using count.")
                is_count_plot = True
                group_by_cols = [current_x_col]
                if current_color_col: group_by_cols.append(current_color_col)
                plot_df = plot_df.groupby(group_by_cols, observed=True, as_index=False).size().rename(columns={'size': temp_count_col_name})
                y_col_for_plot = temp_count_col_name
                y_axis_label = y_axis_label or "Count"
            elif y_col_for_plot: # y_col is valid and numeric
                y_axis_label = y_axis_label or y_col_for_plot
                if current_color_col: # Aggregate if color is also present
                     plot_df = plot_df.groupby([current_x_col, current_color_col], observed=True, as_index=False)[y_col_for_plot].sum()

            plot_df, limited_x, suffix_x = _apply_category_limits(plot_df, config, current_x_col, y_col_for_plot)
            if limited_x and suffix_x: additional_title_suffix += suffix_x
            
            if current_color_col and current_color_col in plot_df.columns:
                plot_df, limited_color, suffix_color = _apply_category_limits(plot_df, config, current_color_col, y_col_for_plot)
                if limited_color and suffix_color: additional_title_suffix += (", " if additional_title_suffix.strip() else "") + suffix_color.strip()

            barmode = 'group' if current_color_col else 'relative'
            if config.bar_style == "stacked": barmode = 'stack'
            elif config.bar_style == "grouped": barmode = 'group'
            
            is_faceted_plot = bool(config.facet_column or config.facet_row)
            facet_wrap, num_facets = get_facet_wrap_and_count(plot_df, config.facet_column or config.facet_row) # Check both
            dynamic_height = calculate_dynamic_height(
                num_items=num_facets,
                wrap_cols=facet_wrap,
                is_faceted=is_faceted_plot,
                item_height=200 if is_faceted_plot else 450
            )
            
            fig = px.bar(
                plot_df, x=current_x_col, y=y_col_for_plot, color=current_color_col,
                barmode=barmode, facet_col=config.facet_column, facet_row=config.facet_row, facet_col_wrap=facet_wrap,
                height=dynamic_height
            )
            num_x_cats = plot_df[current_x_col].nunique()
            fig.update_xaxes(type='category', categoryorder='total descending' if (is_count_plot or (y_col_for_plot and not current_color_col)) else 'trace',
                             tickangle=-45 if num_x_cats > 8 else 0)

        elif actual_plot_type in ["pie", "doughnut"]:
            names_col = config.x_column
            values_col = config.y_column
            if not names_col or names_col not in plot_df.columns:
                print(f"UTILS_PX_PIE_ERROR: Pie/Doughnut needs 'names' column (from x_column)."); return None

            if not final_title_base: final_title_base = f"Proportion of {names_col}"
            
            if not values_col or (values_col in plot_df.columns and not pd.api.types.is_numeric_dtype(plot_df[values_col])):
                if values_col: print(f"UTILS_PX_PIE_WARN: Values col '{values_col}' non-numeric. Using counts.")
                temp_pie_count_col = '_pie_count_'
                plot_df = plot_df.groupby(names_col, observed=True, as_index=False).size().rename(columns={'size': temp_pie_count_col})
                values_col = temp_pie_count_col
            
            plot_df, limited_pie, suffix_pie = _apply_category_limits(plot_df, config, names_col, values_col)
            if limited_pie and suffix_pie: additional_title_suffix += suffix_pie

            fig = px.pie(
                plot_df, names=names_col, values=values_col,
                hole=0.4 if actual_plot_type == "doughnut" else 0
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')

        elif actual_plot_type == "histogram":
            if not config.x_column or config.x_column not in plot_df.columns or not pd.api.types.is_numeric_dtype(plot_df[config.x_column]):
                print(f"UTILS_PX_HIST_ERROR: Histogram needs numeric x_column ('{config.x_column}')."); return None
            
            if not final_title_base: final_title_base = f"Histogram of {config.x_column}"
            current_color_col_hist = config.color_by_column if config.color_by_column and config.color_by_column in plot_df.columns else None

            if current_color_col_hist:
                 plot_df, limited, suffix = _apply_category_limits(plot_df, config, current_color_col_hist, None)
                 if limited and suffix: additional_title_suffix += suffix
            
            facet_wrap, num_facets = get_facet_wrap_and_count(plot_df, config.facet_column)
            dynamic_height = calculate_dynamic_height(num_items=num_facets, wrap_cols=facet_wrap)

            fig = px.histogram(
                plot_df, x=config.x_column, color=current_color_col_hist, nbins=config.bins, marginal="rug",
                facet_col=config.facet_column, facet_row=config.facet_row, facet_col_wrap=facet_wrap,
                height=dynamic_height
            )
            fig.update_layout(bargap=0.1, yaxis_title_text=config.ylabel or "Frequency")


        elif actual_plot_type == "boxplot":
            if not config.y_column or config.y_column not in plot_df.columns or not pd.api.types.is_numeric_dtype(plot_df[config.y_column]):
                print(f"UTILS_PX_BOX_ERROR: Boxplot needs numeric y_column ('{config.y_column}')."); return None
            
            current_x_col_box = config.x_column if config.x_column and config.x_column in plot_df.columns else None
            current_color_col_box = config.color_by_column if config.color_by_column and config.color_by_column in plot_df.columns else None

            if not final_title_base:
                final_title_base = f"Boxplot of {config.y_column}"
                if current_x_col_box: final_title_base += f" by {current_x_col_box}"

            if current_x_col_box and not pd.api.types.is_numeric_dtype(plot_df[current_x_col_box]):
                 plot_df, limited, suffix = _apply_category_limits(plot_df, config, current_x_col_box, config.y_column)
                 if limited and suffix: additional_title_suffix += suffix
            
            if current_color_col_box:
                 plot_df, limited, suffix = _apply_category_limits(plot_df, config, current_color_col_box, config.y_column)
                 if limited and suffix: additional_title_suffix += (", " if additional_title_suffix.strip() else "") + suffix.strip()
            
            facet_wrap, num_facets = get_facet_wrap_and_count(plot_df, config.facet_column)
            dynamic_height = calculate_dynamic_height(num_items=num_facets, wrap_cols=facet_wrap)

            fig = px.box(
                plot_df, x=current_x_col_box, y=config.y_column, color=current_color_col_box,
                points="outliers", facet_col=config.facet_column, facet_row=config.facet_row, facet_col_wrap=facet_wrap,
                height=dynamic_height
            )
            fig.update_xaxes(type='category' if current_x_col_box and not pd.api.types.is_numeric_dtype(plot_df[current_x_col_box]) else 'auto')

        elif actual_plot_type == "scatter":
            if not config.x_column or config.x_column not in plot_df.columns or \
               not config.y_column or config.y_column not in plot_df.columns or \
               not pd.api.types.is_numeric_dtype(plot_df[config.x_column]) or \
               not pd.api.types.is_numeric_dtype(plot_df[config.y_column]):
                print(f"UTILS_PX_SCATTER_ERROR: Scatter needs numeric x ('{config.x_column}') & y ('{config.y_column}')."); return None

            current_color_col_scatter = config.color_by_column if config.color_by_column and config.color_by_column in plot_df.columns else None
            
            if not final_title_base:
                final_title_base = f"Scatter Plot of {config.y_column} vs. {config.x_column}"

            if current_color_col_scatter:
                 plot_df, limited, suffix = _apply_category_limits(plot_df, config, current_color_col_scatter, None)
                 if limited and suffix: additional_title_suffix += suffix
            
            facet_wrap, num_facets = get_facet_wrap_and_count(plot_df, config.facet_column)
            dynamic_height = calculate_dynamic_height(num_items=num_facets, wrap_cols=facet_wrap)
            
            fig = px.scatter(
                plot_df, x=config.x_column, y=config.y_column, color=current_color_col_scatter,
                marginal_y="violin", marginal_x="box",
                facet_col=config.facet_column, facet_row=config.facet_row, facet_col_wrap=facet_wrap,
                height=dynamic_height
            )

        elif actual_plot_type == "kde": # Using px.violin for 1D KDE-like, or density_contour for 2D
             if not config.x_column or config.x_column not in plot_df.columns or not pd.api.types.is_numeric_dtype(plot_df[config.x_column]):
                print(f"UTILS_PX_KDE_ERROR: KDE needs numeric x_column ('{config.x_column}')."); return None
            
             if not final_title_base: final_title_base = f"KDE Plot of {config.x_column}"
             current_color_col_kde = config.color_by_column if config.color_by_column and config.color_by_column in plot_df.columns else None

             if current_color_col_kde:
                 plot_df, limited, suffix = _apply_category_limits(plot_df, config, current_color_col_kde, None)
                 if limited and suffix: additional_title_suffix += suffix
            
             facet_wrap, num_facets = get_facet_wrap_and_count(plot_df, config.facet_column)
             dynamic_height = calculate_dynamic_height(num_items=num_facets, wrap_cols=facet_wrap)

             # If y_column is provided and numeric, use density_contour for 2D KDE
             if config.y_column and config.y_column in plot_df.columns and pd.api.types.is_numeric_dtype(plot_df[config.y_column]):
                 fig = px.density_contour(plot_df, x=config.x_column, y=config.y_column, color=current_color_col_kde,
                                          marginal_x="rug", marginal_y="histogram",
                                          facet_col=config.facet_column, facet_row=config.facet_row, facet_col_wrap=facet_wrap,
                                          height=dynamic_height)
                 fig.update_layout(yaxis_title_text=config.ylabel or config.y_column)
             else: # Otherwise, use violin plot for a 1D KDE-like visualization
                 fig = px.violin(plot_df, y=config.x_column, color=current_color_col_kde, box=True, points="all",
                                 facet_col=config.facet_column, facet_row=config.facet_row, facet_col_wrap=facet_wrap,
                                 height=dynamic_height)
                 fig.update_layout(yaxis_title_text=config.ylabel or config.x_column, # Y-axis of violin is the data
                                   xaxis_title_text="") # No categorical x for violin in this mode


        # Add other plot types here (heatmap, dot_plot, cumulative_curve, lollipop)
        # For Lollipop, you might still prefer Matplotlib if px.bar with styling isn't sufficient.
        # px.density_heatmap, px.strip, px.ecdf are available.

        else:
            print(f"UTILS_GENERATE_PLOT_WARN: Plot type '{actual_plot_type}' not fully implemented with Plotly Express.")
            # Matplotlib fallback (Simplified - consider if really needed or if px can cover)
            # import matplotlib.pyplot as plt
            # import seaborn as sns
            # try:
            #     # ... your existing Matplotlib code for specific plots like lollipop ...
            #     # if actual_plot_type == "lollipop": ...
            #     # fig_mpl, ax_mpl = plt.subplots(...)
            #     # ... plotting ...
            #     # plotly_fig_obj = plotly.tools.mpl_to_plotly(fig_mpl)
            #     # plt.close(fig_mpl)
            #     # fig = plotly_fig_obj
            #     pass # Placeholder
            # except Exception as e_mpl:
            #     print(f"UTILS_MPL_FALLBACK_ERROR: {e_mpl}");
            #     return None
            return None # If not handled by px and no MPL fallback

        # Final layout updates for all plots
        if fig:
            fig.update_layout(
                title_text=final_title_base + additional_title_suffix,
                xaxis_title_text=config.xlabel or config.x_column, # This might be overridden by specific plot logic
                # yaxis_title_text is usually set by specific plot logic already
                legend_title_text=config.color_by_column # This also might be overridden
            )
            # Clean up faceted subplot titles
            if config.facet_column or config.facet_row:
                fig.for_each_xaxis(lambda axis: axis.update(title=None))
                fig.for_each_yaxis(lambda axis: axis.update(title=None))
            
            return pio.to_json(fig)
        else:
            print(f"UTILS_GENERATE_PLOT_ERROR: Figure object was not created for '{actual_plot_type}'.")
            return None

    except Exception as e_px_global:
        print(f"UTILS_PX_PLOT_GLOBAL_ERROR: Unhandled error during Plotly Express generation for '{actual_plot_type}': {e_px_global}")
        traceback.print_exc()
        return None

def calculate_age_from_dob(df: pd.DataFrame, dob_column_name: str) -> Optional[pd.Series]:
    print(f"UTILS_DOB: Parsing DOB from column: '{dob_column_name}'")
    if dob_column_name not in df.columns:
        print(f"UTILS_DOB_ERR: Column '{dob_column_name}' not found in DataFrame."); return None
    
    try:
        # Attempt standard parsing
        dob_s = pd.to_datetime(df[dob_column_name], errors='coerce')
        
        # If all are NaT or many are NaT, try with dayfirst=True
        # This is a heuristic; more sophisticated date parsing might be needed for very mixed formats
        if dob_s.isnull().all() or (dob_s.isnull().sum() > len(df) * 0.5 and len(df) > 0) : # If all or >50% are NaT
            print(f"UTILS_DOB_INFO: Standard parsing of '{dob_column_name}' yielded many NaTs. Trying with dayfirst=True.")
            dob_s_dayfirst = pd.to_datetime(df[dob_column_name], dayfirst=True, errors='coerce')
            # Use dayfirst result if it has more valid (non-NaT) dates
            if dob_s_dayfirst.notna().sum() > dob_s.notna().sum():
                dob_s = dob_s_dayfirst
                print(f"UTILS_DOB_INFO: Used dayfirst=True parsing for '{dob_column_name}'.")

        if dob_s.isnull().all(): # Check again after potential dayfirst attempt
            print(f"UTILS_DOB_ERR: Cannot parse any valid dates from '{dob_column_name}' even with dayfirst=True attempt."); return None
        
        # Filter out future dates of birth, as they are invalid for age calculation
        today_dt = datetime.today()
        valid_dob_s = dob_s[dob_s <= pd.Timestamp(today_dt)]
        if len(valid_dob_s) < len(dob_s):
            print(f"UTILS_DOB_WARN: Filtered out {len(dob_s) - len(valid_dob_s)} future DOBs from '{dob_column_name}'.")
        
        if valid_dob_s.empty and not dob_s.isnull().all(): # All DOBs were future dates or unparseable
             print(f"UTILS_DOB_ERR: No valid past/present DOBs found in '{dob_column_name}' for age calculation."); return None
        elif valid_dob_s.empty and dob_s.isnull().all(): # All were unparseable initially
             print(f"UTILS_DOB_ERR: All DOBs were unparseable in '{dob_column_name}'."); return None


        # Calculate age based on valid DOBs
        # Ensure we are operating on a Series that aligns with the original DataFrame's index
        age_series = pd.Series(index=df.index, dtype='Int64')

        # Calculate age for valid, non-NaT DOBs
        # Ensure 'today_dt' is timezone-naive if 'valid_dob_s.dt' is timezone-naive, or vice-versa
        if valid_dob_s.dt.tz is not None:
            today_ts = pd.Timestamp(today_dt, tz=valid_dob_s.dt.tz)
        else:
            today_ts = pd.Timestamp(today_dt)


        # Vectorized age calculation
        # Subtracting years
        age = today_ts.year - valid_dob_s.dt.year
        
        # Adjust for month/day for those whose birthday hasn't occurred yet this year
        # Condition for adjustment: (birth_month > current_month) OR (birth_month == current_month AND birth_day > current_day)
        month_day_adjustment_needed = \
            (valid_dob_s.dt.month > today_ts.month) | \
            ((valid_dob_s.dt.month == today_ts.month) & (valid_dob_s.dt.day > today_ts.day))
        
        age_adjusted = age - month_day_adjustment_needed.astype(int)
        
        # Assign calculated ages to the full-indexed series
        age_series.loc[valid_dob_s.index] = age_adjusted
        
        if age_series.notnull().any():
            stats = age_series.dropna().astype(float) 
            if not stats.empty: 
                print(f"UTILS_DOB: Age stats for '{dob_column_name}': min={stats.min()}, max={stats.max()}, mean={stats.mean():.1f}, NaNs (unparseable/future)={age_series.isnull().sum()}")
        else:
            print(f"UTILS_DOB_INFO: All ages calculated as NaT for '{dob_column_name}'. Original NaNs in DOB col: {df[dob_column_name].isnull().sum()}, Unparseable/Future: {age_series.isnull().sum()}")
            
        return age_series
    except Exception as e: 
        print(f"UTILS_DOB_EXC: Exception during age calculation for '{dob_column_name}': {e}"); traceback.print_exc(); return None
# --- END OF FILE utils.py ---