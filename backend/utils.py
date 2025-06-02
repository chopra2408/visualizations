import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from typing import Optional 
from models import PlotConfig

def generate_plot_from_config(df: pd.DataFrame, config: PlotConfig) -> Optional[bytes]:
    print(f"--- Generating plot with config: {config.model_dump_json(indent=2)}") 
    print(f"DataFrame dtypes:\n{df.dtypes}")
    if df.empty:
        print("DataFrame is empty. Cannot generate plot.")
        return None

    plt.figure(figsize=(10, 6))  
    try: 
        if config.x_column and config.x_column not in df.columns:
            print(f"X-column '{config.x_column}' not found in DataFrame.")
            return None  
        if config.y_column and config.y_column not in df.columns:
            print(f"Y-column '{config.y_column}' not found in DataFrame.")
            return None
        if config.color_by_column and config.color_by_column not in df.columns:
            print(f"Color-by-column '{config.color_by_column}' not found in DataFrame.")
            return None

        plot_generated = False
        if config.plot_type == "bar":
            if config.x_column:
                if config.y_column:  
                    sns.barplot(data=df, x=config.x_column, y=config.y_column, hue=config.color_by_column, estimator=sum) # or mean, etc.
                else:  
                    sns.countplot(data=df, x=config.x_column, hue=config.color_by_column)
                plot_generated = True
            else:
                print("Bar plot requires an x_column.")
        
        elif config.plot_type == "histogram":
            if config.x_column:
                sns.histplot(data=df, x=config.x_column, hue=config.color_by_column, bins=config.bins or 'auto', kde=True) # Added kde=True for distribution
                plot_generated = True
            else:
                print("Histogram requires an x_column.")

        elif config.plot_type == "kde":  
            if config.x_column:
                sns.kdeplot(data=df, x=config.x_column, hue=config.color_by_column, fill=True)
                plot_generated = True
            else:
                print("KDE plot requires an x_column.")

        elif config.plot_type == "scatter":
            if config.x_column and config.y_column:
                sns.scatterplot(data=df, x=config.x_column, y=config.y_column, hue=config.color_by_column)
                plot_generated = True
            else:
                print("Scatter plot requires x_column and y_column.")

        elif config.plot_type == "line":
            if config.x_column and config.y_column: 
                sns.lineplot(data=df, x=config.x_column, y=config.y_column, hue=config.color_by_column)
                plot_generated = True
            else:
                print("Line plot requires x_column and y_column.")
        
        elif config.plot_type == "boxplot":
            if config.x_column:  
                sns.boxplot(data=df, x=config.x_column, y=config.y_column, hue=config.color_by_column)
                plot_generated = True
            elif config.y_column: 
                sns.boxplot(data=df, y=config.y_column)
                plot_generated = True
            else:
                print("Boxplot requires at least an x_column (for categories) or a y_column (for numerical data).")

        if not plot_generated:
            print(f"Plot type '{config.plot_type}' not recognized or prerequisites not met.")
            plt.close()  
            return None
 
        plt.title(config.title or f"{config.plot_type.capitalize()} of {config.x_column or ''}{' vs ' + config.y_column if config.y_column else ''}")
        if config.xlabel: plt.xlabel(config.xlabel)
        elif config.x_column: plt.xlabel(config.x_column)
        if config.ylabel: plt.ylabel(config.ylabel)
        elif config.y_column: plt.ylabel(config.y_column)
        
        plt.xticks(rotation=45, ha='right')  
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()  
        buf.seek(0)
        return buf.getvalue()

    except KeyError as e:
        print(f"KeyError during plot generation: {e}. Check column names.")
        plt.close()
        return None
    except Exception as e:
        print(f"Error generating plot from config: {e}")
        import traceback
        traceback.print_exc()
        plt.close() 
        return None