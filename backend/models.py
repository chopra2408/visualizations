from pydantic import BaseModel, Field
from typing import Optional, List, Literal

class QueryRequest(BaseModel):
    session_id: str
    query: str

class FileInfo(BaseModel):
    session_id: str
    filename: str
    columns: List[str]
    df_head: str

class PlotConfig(BaseModel):
    plot_type: Literal[
        "bar", "line", "scatter", "histogram", "boxplot", "kde", "auto_categorical",
        "heatmap", "dot_plot", "cumulative_curve", "lollipop", "pie", "doughnut"
    ] = Field(
        description="The type of plot to generate. 'auto_categorical' defaults to 'bar'."
    )
    x_column: Optional[str] = Field(None, description="Column for the X-axis or for labels/names in pie charts.")
    y_column: Optional[str] = Field(None, description="Column for the Y-axis or for values in pie/bar charts.")
    color_by_column: Optional[str] = Field(None, description="Column to use for coloring/hue/stacking.")
    title: Optional[str] = Field(None, description="Suggested title for the plot.")
    xlabel: Optional[str] = Field(None, description="Suggested label for the X-axis.")
    ylabel: Optional[str] = Field(None, description="Suggested label for the Y-axis.")
    bins: Optional[int] = Field(None, description="Number of bins (for histogram).")
    bar_style: Optional[Literal["stacked", "grouped", "auto"]] = Field("auto", description="For bar charts, specify if it should be stacked or grouped. 'auto' lets the system decide.")

    # New fields for improved readability and handling high cardinality:
    top_n_categories: Optional[int] = Field(None, description="If specified, plot only the top N categories for the x_column or color_by_column, based on y_column values or frequency/count.")
    top_n_metric: Optional[Literal["sum", "mean", "median", "count"]] = Field("sum", description="Metric to use for determining top N categories when y_column is numeric. Use 'count' if y_column is not specified or not numeric.")
    facet_column: Optional[str] = Field(None, description="Column to use for faceting (creating small multiples based on this column's categories).")
    facet_row: Optional[str] = Field(None, description="Column to use for faceting rows (creating small multiples based on this column's categories).")
    aggregate_by_color_col_method: Optional[Literal["mean", "median", "sum"]] = Field(None, description="For line plots with many series in color_by_column: aggregate y_column by this method (e.g., 'mean') to show a single summary line instead of many individual lines.")
    limit_categories_auto: Optional[bool] = Field(False, description="If true, automatically apply a reasonable limit (e.g. top 7-10) if color_by_column or x_column for categorical plots has too many unique values and top_n_categories is not set. The system will choose a sensible default for N.")


class AgentResponse(BaseModel):
    response_type: str
    content: Optional[str] = None
    plotly_fig_json: Optional[str] = None
    plot_config_json: Optional[str] = None
    plot_insights: Optional[str] = None
    thinking_log_str: Optional[str] = None
    error: Optional[str] = None