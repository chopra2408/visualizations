# # --- START OF FILE models.py ---

# from pydantic import BaseModel, Field
# from typing import Optional, List, Literal

# class QueryRequest(BaseModel):
#     session_id: str
#     query: str

# class FileInfo(BaseModel):
#     session_id: str
#     filename: str
#     columns: List[str]
#     df_head: str

# class PlotConfig(BaseModel):
#     plot_type: Literal[
#         "bar", "line", "scatter", "histogram", "boxplot", "kde", "auto_categorical" # New "auto_categorical"
#     ] = Field(
#         description="The type of plot to generate. 'bar' can be stacked or grouped. 'auto_categorical' lets the system choose the best plot for categorical data (e.g. bar, possibly stacked)."
#     )
#     x_column: Optional[str] = Field(None, description="Column for the X-axis. Required for most plots.")
#     y_column: Optional[str] = Field(None, description="Column for the Y-axis (e.g., for scatter, line, bar if not counts).")
#     color_by_column: Optional[str] = Field(None, description="Column to use for coloring/hue/stacking.")
#     title: Optional[str] = Field(None, description="Suggested title for the plot.")
#     xlabel: Optional[str] = Field(None, description="Suggested label for the X-axis.")
#     ylabel: Optional[str] = Field(None, description="Suggested label for the Y-axis.")
#     bins: Optional[int] = Field(None, description="Number of bins (for histogram). Default is 'auto' if not provided or invalid.")
#     # Potentially add:
#     bar_style: Optional[Literal["stacked", "grouped", "auto"]] = Field("auto", description="For bar charts, specify if it should be stacked or grouped. 'auto' lets the system decide.")

# class AgentResponse(BaseModel):
#     response_type: str
#     content: Optional[str] = None
#     # plot_image_bytes: Optional[str] = None # OLD
#     plotly_fig_json: Optional[str] = None   # NEW for Plotly JSON
#     plot_config_json: Optional[str] = None 
#     plot_insights: Optional[str] = None
#     thinking_log_str: Optional[str] = None
#     error: Optional[str] = None

# --- START OF FILE models.py ---

# --- START OF FILE models.py ---

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
        "bar", "line", "scatter", "histogram", "boxplot", "kde", "auto_categorical" # New "auto_categorical"
    ] = Field(
        description="The type of plot to generate. 'bar' can be stacked or grouped. 'auto_categorical' lets the system choose the best plot for categorical data (e.g. bar, possibly stacked)."
    )
    x_column: Optional[str] = Field(None, description="Column for the X-axis. Required for most plots.")
    y_column: Optional[str] = Field(None, description="Column for the Y-axis (e.g., for scatter, line, bar if not counts).")
    color_by_column: Optional[str] = Field(None, description="Column to use for coloring/hue/stacking.")
    title: Optional[str] = Field(None, description="Suggested title for the plot.")
    xlabel: Optional[str] = Field(None, description="Suggested label for the X-axis.")
    ylabel: Optional[str] = Field(None, description="Suggested label for the Y-axis.")
    bins: Optional[int] = Field(None, description="Number of bins (for histogram). Default is 'auto' if not provided or invalid.")
    bar_style: Optional[Literal["stacked", "grouped", "auto"]] = Field("auto", description="For bar charts, specify if it should be stacked or grouped. 'auto' lets the system decide.")

class AgentResponse(BaseModel):
    response_type: str
    content: Optional[str] = None
    # plot_image_bytes: Optional[str] = None # OLD
    plotly_fig_json: Optional[str] = None   # NEW for Plotly JSON
    plot_config_json: Optional[str] = None
    plot_insights: Optional[str] = None
    thinking_log_str: Optional[str] = None
    error: Optional[str] = None
# --- END OF FILE models.py ---