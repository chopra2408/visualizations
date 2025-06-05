 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from datetime import datetime
from backend.models import PlotConfig # Assuming models.py is in backend folder
import traceback
import numpy as np
import plotly.tools
import plotly.io as pio # For converting to JSON
import plotly.express as px # For native Plotly plots
import plotly.graph_objects as go # Import graph_objects for Layout

def _reshape_for_stacked_bar(df: pd.DataFrame, x_col: str, y_col: Optional[str], color_by_col: str, aggfunc=np.sum) -> Optional[pd.DataFrame]:
    # (This function is assumed correct and remains unchanged from your version)
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
        return pivot_df
    except Exception as e: print(f"UTILS_RESHAPE_CRITICAL_ERROR: {e}"); traceback.print_exc(); return None

def generate_plot_from_config(df: pd.DataFrame, config: PlotConfig) -> Optional[str]:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("UTILS_GENERATE_PLOT: NATIVE PLOTLY + MPL w/ TRACE DATA ALIGNMENT V6 (Native PX Bar)")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"UTILS_GENERATE_PLOT: Received config: {config.model_dump_json(indent=0)}")

    if df.empty: print("UTILS_GENERATE_PLOT_ERROR: DataFrame is empty."); return None

    actual_plot_type = config.plot_type
    if config.plot_type == "auto_categorical": actual_plot_type = "bar" # auto_categorical defaults to bar

    # --- NATIVE PLOTLY PATHS ---
    if actual_plot_type == "boxplot":
        print("UTILS_NATIVE_PLOTLY: BOXPLOT with Plotly Express.")
        try:
            if not config.y_column or config.y_column not in df.columns or not pd.api.types.is_numeric_dtype(df[config.y_column]):
                print(f"UTILS_NATIVE_PLOTLY_BOX_ERROR: Boxplot needs valid numeric y_column ('{config.y_column}')."); return None
            
            current_x_col = config.x_column if config.x_column and config.x_column in df.columns else None
            current_color_col = config.color_by_column if config.color_by_column and config.color_by_column in df.columns else None
            
            xaxis_type_setting = 'auto'
            if current_x_col:
                if pd.api.types.is_string_dtype(df[current_x_col]) or pd.api.types.is_categorical_dtype(df[current_x_col]):
                    xaxis_type_setting = 'category'
                elif pd.api.types.is_numeric_dtype(df[current_x_col]) and df[current_x_col].nunique() <= 50:
                     xaxis_type_setting = 'category' 
            
            final_main_title = config.title or f"Boxplot of {config.y_column}{' by ' + current_x_col if current_x_col else ''}"
            final_x_label = config.xlabel or current_x_col or ""
            final_y_label = config.ylabel or config.y_column
            final_legend_title = current_color_col 

            plotly_fig_native = px.box(
                df, x=current_x_col, y=config.y_column, color=current_color_col,
                points="outliers"
            )
            
            plotly_fig_native.update_layout(
                title_text=final_main_title,
                xaxis_title_text=final_x_label,
                yaxis_title_text=final_y_label,
                xaxis_type=xaxis_type_setting,
                legend_title_text=final_legend_title if current_color_col else None
            )
            return pio.to_json(plotly_fig_native)
        except Exception as e_native_px_box: print(f"UTILS_NATIVE_PLOTLY_BOX_ERROR: {e_native_px_box}"); traceback.print_exc(); return None
    
    elif actual_plot_type == "histogram":
        print("UTILS_NATIVE_PLOTLY: HISTOGRAM with Plotly Express.")
        try:
            if not config.x_column or config.x_column not in df.columns or not pd.api.types.is_numeric_dtype(df[config.x_column]):
                print(f"UTILS_NATIVE_PLOTLY_HIST_ERROR: Histogram needs valid numeric x_column ('{config.x_column}')."); return None

            current_color_col = config.color_by_column if config.color_by_column and config.color_by_column in df.columns else None

            final_main_title = config.title or f"Histogram of {config.x_column}{' by ' + current_color_col if current_color_col else ''}"
            final_x_label = config.xlabel or config.x_column
            final_y_label = config.ylabel or "Frequency"
            final_legend_title = current_color_col

            plotly_fig_native = px.histogram(
                df, x=config.x_column, color=current_color_col, marginal="rug", nbins=config.bins
            )
            
            plotly_fig_native.update_layout(
                title_text=final_main_title,
                xaxis_title_text=final_x_label,
                yaxis_title_text=final_y_label,
                bargap=0.1 if not current_color_col else 0.2, # Default for histogram
                legend_title_text=final_legend_title if current_color_col else None
            )
            return pio.to_json(plotly_fig_native)
        except Exception as e_native_px_hist: print(f"UTILS_NATIVE_PLOTLY_HIST_ERROR: {e_native_px_hist}"); traceback.print_exc(); return None

    elif actual_plot_type == "bar":
        print("UTILS_NATIVE_PLOTLY: BAR CHART with Plotly Express.")
        try:
            if not config.x_column or config.x_column not in df.columns:
                print(f"UTILS_NATIVE_PLOTLY_BAR_ERROR: Bar chart needs valid x_column ('{config.x_column}').")
                return None

            current_x_col = config.x_column
            current_y_col_config = config.y_column
            current_color_col = config.color_by_column if config.color_by_column and config.color_by_column in df.columns else None
            
            num_x_categories = df[current_x_col].nunique()
            
            plot_df = df.copy() 
            y_axis_label_final = config.ylabel 
            y_col_for_plot = current_y_col_config
            is_count_plot = False

            if not y_col_for_plot: 
                print(f"UTILS_NATIVE_PLOTLY_BAR: Y-column not specified, creating a count plot for x='{current_x_col}'.")
                is_count_plot = True
                count_col_name = '_count_'
                group_by_cols = [current_x_col]
                if current_color_col:
                    group_by_cols.append(current_color_col)
                
                temp_counts_df = df.groupby(group_by_cols, observed=True).size().reset_index(name=count_col_name)
                plot_df = temp_counts_df 
                y_col_for_plot = count_col_name
                if not y_axis_label_final: y_axis_label_final = "Count"
            
            elif current_y_col_config and current_y_col_config in df.columns:
                if not pd.api.types.is_numeric_dtype(df[current_y_col_config]):
                    print(f"UTILS_NATIVE_PLOTLY_BAR_WARN: Specified Y-column '{current_y_col_config}' is not numeric. Fallback to count plot on X.")
                    is_count_plot = True
                    count_col_name = '_count_fallback_'
                    group_by_cols = [current_x_col]
                    if current_color_col: group_by_cols.append(current_color_col)
                    temp_counts_df = df.groupby(group_by_cols, observed=True).size().reset_index(name=count_col_name)
                    plot_df = temp_counts_df
                    y_col_for_plot = count_col_name
                    if not y_axis_label_final: y_axis_label_final = "Count"
                else:
                    if not y_axis_label_final: y_axis_label_final = y_col_for_plot
            else: 
                print(f"UTILS_NATIVE_PLOTLY_BAR_ERROR: Specified y_column '{current_y_col_config}' not in DataFrame or invalid. Cannot plot.")
                return None

            barmode_setting = 'group' 
            if config.bar_style == "stacked":
                barmode_setting = 'stack'
            elif config.bar_style == "grouped":
                barmode_setting = 'group'
            elif current_color_col: 
                barmode_setting = 'group' 
            else: 
                barmode_setting = 'relative'

            final_main_title = config.title or f"Bar Chart of {current_x_col}{(' by ' + current_color_col) if current_color_col else ''}"
            final_x_label = config.xlabel or current_x_col
            final_legend_title = current_color_col 

            plotly_fig_native = px.bar(
                plot_df, 
                x=current_x_col, 
                y=y_col_for_plot, 
                color=current_color_col,
                barmode=barmode_setting
            )
            
            tickangle_setting = 0
            categoryorder_setting = 'trace' 
            if is_count_plot:
                 # For count plots, sorting by frequency (total count) is often best
                categoryorder_setting = 'total descending'
            
            # Update X-axis
            x_axis_layout = {
                "title_text": final_x_label,
                "type": 'category', 
                "categoryorder": categoryorder_setting
            }

            if num_x_categories > 8 and num_x_categories <= 20: # Adjusted threshold
                tickangle_setting = -45
            elif num_x_categories > 20: # Adjusted threshold
                tickangle_setting = -60 # Slightly less extreme than -90 initially
            
            if tickangle_setting != 0:
                x_axis_layout["tickangle"] = tickangle_setting
            
            plotly_fig_native.update_layout(
                title_text=final_main_title,
                xaxis=x_axis_layout,
                yaxis_title_text=y_axis_label_final,
                legend_title_text=final_legend_title if current_color_col else None,
                bargap=0.2 # Default gap for bar charts
            )
            return pio.to_json(plotly_fig_native)
        except Exception as e_native_px_bar:
            print(f"UTILS_NATIVE_PLOTLY_BAR_ERROR: {e_native_px_bar}")
            traceback.print_exc()
            return None

    # --- MATPLOTLIB/SEABORN PATH FOR OTHER PLOTS (KDE, Line, Scatter) ---
    print(f"UTILS_MPL_PLOT: Matplotlib path for plot type: '{actual_plot_type}'")
    fig_width=10; fig_height=6; xtick_rotation=0; xtick_ha='center'; xtick_fontsize=10
    final_rect_bottom_margin=0.12; final_rect_right_margin=0.95; num_x_categories=0
    
    current_x_col_mpl = config.x_column
    current_y_col_mpl = config.y_column
    current_color_col_mpl = config.color_by_column

    # X-column is essential for these Matplotlib plot types
    if actual_plot_type in ["kde", "line", "scatter"]: # Bar removed from this check
        if current_x_col_mpl and current_x_col_mpl in df.columns:
            num_x_categories = df[current_x_col_mpl].nunique()
        else:
            print(f"UTILS_MPL_ERR: X-col missing/not found for '{actual_plot_type}'."); return None
    
    # Specific sizing adjustments (Bar chart sizing removed from here)
    if actual_plot_type == "kde":
        fig_width,fig_height,final_rect_bottom_margin=12,7,0.15
    elif actual_plot_type == "line" and num_x_categories > 15: # Example for line if many points
        xtick_rotation = 30
        final_rect_bottom_margin = 0.18


    print(f"UTILS_MPL_PLOT: type='{actual_plot_type}', fig_w={fig_width}, fig_h={fig_height}, rot={xtick_rotation}")
    plt.style.use('seaborn-v0_8-whitegrid'); fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    try:
        if current_y_col_mpl and current_y_col_mpl not in df.columns: print(f"UTILS_MPL_ERR: Y-col '{current_y_col_mpl}' not found."); plt.close(fig); return None
        if current_color_col_mpl and current_color_col_mpl not in df.columns: print(f"UTILS_MPL_ERR: Color-col '{current_color_col_mpl}' not found."); plt.close(fig); return None
        plot_generated = False
        
        # Matplotlib plotting logic - Bar chart plotting removed from here
        if actual_plot_type == "kde":
            if not pd.api.types.is_numeric_dtype(df[current_x_col_mpl]): print(f"UTILS_MPL_ERR: KDE needs numeric X-col."); plt.close(fig);return None
            sns.kdeplot(data=df, x=current_x_col_mpl, hue=current_color_col_mpl, fill=False, linewidth=2, ax=ax); plot_generated=True
        elif actual_plot_type == "line": 
            df_s=df.copy()
            if current_x_col_mpl in df_s.columns and (pd.api.types.is_datetime64_any_dtype(df_s[current_x_col_mpl])or pd.api.types.is_numeric_dtype(df_s[current_x_col_mpl])): df_s=df_s.sort_values(by=current_x_col_mpl)
            if not current_y_col_mpl:
                cnt_col='_g_cnt_mpl';
                if pd.api.types.is_numeric_dtype(df_s[current_x_col_mpl]): sns.lineplot(data=df_s,x=current_x_col_mpl,y=df_s.index,hue=current_color_col_mpl,marker='o',ax=ax);ax.set_ylabel(config.ylabel or "Index")
                else: 
                    if current_color_col_mpl: cnt_data=df_s.groupby([current_x_col_mpl,current_color_col_mpl],observed=False).size().reset_index(name=cnt_col); sns.lineplot(data=cnt_data,x=current_x_col_mpl,y=cnt_col,hue=current_color_col_mpl,marker='o',ax=ax)
                    else: cnt_data=df_s.groupby(current_x_col_mpl,observed=False).size().reset_index(name=cnt_col); sns.lineplot(data=cnt_data,x=current_x_col_mpl,y=cnt_col,marker='o',ax=ax)
                    ax.set_ylabel(config.ylabel or "Count")
            else: sns.lineplot(data=df_s,x=current_x_col_mpl,y=current_y_col_mpl,hue=current_color_col_mpl,marker='o',ax=ax)
            plot_generated=True
        elif actual_plot_type == "scatter": 
            if not current_y_col_mpl: print(f"UTILS_MPL_ERR: Scatter needs Y-col."); plt.close(fig);return None
            if not pd.api.types.is_numeric_dtype(df[current_x_col_mpl]) or not pd.api.types.is_numeric_dtype(df[current_y_col_mpl]):
                 print(f"UTILS_MPL_ERR: Scatter needs numeric X and Y columns."); plt.close(fig);return None
            sns.scatterplot(data=df,x=current_x_col_mpl,y=current_y_col_mpl,hue=current_color_col_mpl,ax=ax);plot_generated=True
        
        if not plot_generated:
             print(f"UTILS_MPL_ERR: Plot not generated for type '{actual_plot_type}'. Could be an unhandled type or error in specific plot logic.");
             plt.close(fig); return None

        mpl_def_y_lab="Value"
        # Removed bar-specific default y-label
        if actual_plot_type=='kde': mpl_def_y_lab="Density"
        
        mpl_final_main_title = config.title or f"{actual_plot_type.capitalize()} Plot of {current_x_col_mpl if current_x_col_mpl else 'Data'}"
        mpl_final_x_label = config.xlabel or current_x_col_mpl or ""
        mpl_final_y_label = config.ylabel or current_y_col_mpl or mpl_def_y_lab
        
        ax.set_title(mpl_final_main_title, fontsize=14, wrap=True, pad=20)
        ax.set_xlabel(mpl_final_x_label, fontsize=12)
        ax.set_ylabel(mpl_final_y_label, fontsize=12)
        
        unique_x_categories_for_mpl_ticks = []
        # Bar chart tick handling removed from here
        if actual_plot_type in ["line"] and current_x_col_mpl and current_x_col_mpl in df.columns: # Adjusted this condition
            unique_x_categories_for_mpl_ticks = sorted(df[current_x_col_mpl].dropna().unique())
            if 0 < len(unique_x_categories_for_mpl_ticks) <= 50: 
                if pd.api.types.is_numeric_dtype(df[current_x_col_mpl]):
                    ax.set_xticks(unique_x_categories_for_mpl_ticks)
                    ax.set_xticklabels([str(int(v)) if v == round(v) else f"{v:.2f}" if isinstance(v, float) else str(v) for v in unique_x_categories_for_mpl_ticks], rotation=xtick_rotation, ha=xtick_ha, fontsize=xtick_fontsize)
                else: 
                    ax.set_xticks(range(len(unique_x_categories_for_mpl_ticks))) 
                    ax.set_xticklabels([str(v) for v in unique_x_categories_for_mpl_ticks], rotation=xtick_rotation, ha=xtick_ha, fontsize=xtick_fontsize)
                print(f"UTILS_MPL_PLOT: Set custom MPL ticks for X: '{current_x_col_mpl}'")
            else: 
                ax.tick_params(axis='x', labelsize=xtick_fontsize)
                if xtick_rotation > 0: plt.setp(ax.get_xticklabels(), rotation=xtick_rotation, ha=xtick_ha)
        elif current_x_col_mpl: 
            ax.tick_params(axis='x', labelsize=xtick_fontsize)
            if xtick_rotation > 0: plt.setp(ax.get_xticklabels(), rotation=xtick_rotation, ha=xtick_ha)
        ax.tick_params(axis='y', labelsize=10)

        leg = ax.get_legend(); 
        if leg: leg.remove()
        
        try:
            fig.tight_layout(rect=[0.08, final_rect_bottom_margin, final_rect_right_margin, 0.92])
        except ValueError: 
            fig.tight_layout() 
        
        print("UTILS_MPL_PLOT: Converting Matplotlib fig to Plotly & applying comprehensive label fixes.")
        try:
            plotly_fig_obj: go.Figure = plotly.tools.mpl_to_plotly(fig) 
            
            final_plotly_main_title = ax.get_title()
            final_plotly_x_label = ax.get_xlabel()
            final_plotly_y_label = ax.get_ylabel()
            
            update_layout_dict = {
                'title_text': final_plotly_main_title,
                'xaxis_title_text': final_plotly_x_label,
                'yaxis_title_text': final_plotly_y_label
            }
            
            if current_color_col_mpl:
                update_layout_dict['legend_title_text'] = current_color_col_mpl
            
            # Bar chart specific conversion fixes removed as bar charts are now native Plotly
            if actual_plot_type in ["line"] and current_x_col_mpl and current_x_col_mpl in df.columns:
                if unique_x_categories_for_mpl_ticks: # From earlier logic
                    plotly_categories = [str(v) for v in unique_x_categories_for_mpl_ticks]
                    print(f"UTILS_MPL_PLOT_POST_CONV: Forcing Plotly x-axis: type='category', categoryarray for '{current_x_col_mpl}'.")
                    
                    update_layout_dict['xaxis_type'] = 'category'
                    update_layout_dict['xaxis_categoryorder'] = 'array'
                    update_layout_dict['xaxis_categoryarray'] = plotly_categories
                    update_layout_dict.update({'xaxis_tickvals': None, 'xaxis_ticktext': None, 'xaxis_tickmode': 'auto'})

                    if xtick_rotation != 0: update_layout_dict['xaxis_tickangle'] = -xtick_rotation
                    else: update_layout_dict['xaxis_tickangle'] = None
                    
                    if plotly_fig_obj.data:
                        for trace in plotly_fig_obj.data:
                            if hasattr(trace, 'x') and trace.x is not None:
                                if len(trace.x) == len(plotly_categories):
                                    print(f"UTILS_MPL_PLOT_POST_CONV: Aligning trace x-data with string categories for '{current_x_col_mpl}'.")
                                    trace.x = plotly_categories
                                else:
                                    print(f"UTILS_MPL_PLOT_POST_CONV_WARN: Trace x-data for '{current_x_col_mpl}' (len {len(trace.x)}) "
                                          f"does not directly match categoryarray (len {len(plotly_categories)}). Original trace.x: {trace.x[:5]}")
            
            plotly_fig_obj.update_layout(**update_layout_dict)
            
            print(f"UTILS_PLOTLY_FINAL: Main='{plotly_fig_obj.layout.title.text if plotly_fig_obj.layout.title else 'N/A'}', "
                  f"X='{plotly_fig_obj.layout.xaxis.title.text if plotly_fig_obj.layout.xaxis.title else 'N/A'}', "
                  f"Y='{plotly_fig_obj.layout.yaxis.title.text if plotly_fig_obj.layout.yaxis.title else 'N/A'}'")
            return pio.to_json(plotly_fig_obj)
        except Exception as e_conv: 
            print(f"UTILS_MPL_PLOT_CONV_ERROR: {e_conv}"); traceback.print_exc(); return None
        finally: plt.close(fig)
    except Exception as e_mpl_main: 
        print(f"UTILS_MPL_PLOT_MAIN_ERROR: {e_mpl_main}"); traceback.print_exc(); 
        if 'fig' in locals() and hasattr(fig, 'number'): plt.close(fig)
        return None

def calculate_age_from_dob(df: pd.DataFrame, dob_column_name: str) -> Optional[pd.Series]:
    # (This function remains unchanged from your last working version)
    print(f"UTILS: Parsing DOB: '{dob_column_name}'")
    if dob_column_name not in df.columns: print(f"UTILS_DOB_ERR: Col not found."); return None
    try:
        dob_s = pd.to_datetime(df[dob_column_name], errors='coerce')
        if dob_s.isnull().all():
            dob_s_df = pd.to_datetime(df[dob_column_name], dayfirst=True, errors='coerce')
            if dob_s_df.notna().sum() > 0 and (dob_s.notna().sum()==0 or dob_s_df.notna().sum()>dob_s.notna().sum()): dob_s=dob_s_df; print("UTILS: Used dayfirst parse for DOB.")
        if dob_s.isnull().all(): print(f"UTILS_DOB_ERR: Cannot parse dates from '{dob_column_name}'."); return None
        if dob_s.notnull().sum()==0: return None # Should be redundant due to previous check
        today=pd.Timestamp(datetime.today()); age=today.year-dob_s.dt.year
        # Account for month/day for exact age
        mask=(dob_s.dt.month>today.month)|((dob_s.dt.month==today.month)&(dob_s.dt.day>today.day))
        age=age-mask.astype(int).where(dob_s.notna(),pd.NA); # Ensure NA for NA DOBs
        age=age.astype('Int64') # Use nullable integer type
        if age.notnull().any():
            stats=age.dropna().astype(float) # For stats, drop NA and convert to float
            if not stats.empty: print(f"UTILS: Age stats for '{dob_column_name}': min={stats.min()}, max={stats.max()}, mean={stats.mean():.1f}")
        return age
    except Exception as e: print(f"UTILS_DOB_EXC: {e} for '{dob_column_name}'"); traceback.print_exc(); return None