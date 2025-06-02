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
    plot_type: Literal["bar", "line", "scatter", "histogram", "boxplot", "kde"] = Field(
        description="The type of plot to generate."
    )
    x_column: Optional[str] = Field(None, description="Column for the X-axis. Required for most plots.")
    y_column: Optional[str] = Field(None, description="Column for the Y-axis (e.g., for scatter, line, bar if not counts).")
    # y_columns: Optional[List[str]] = Field(None, description="Multiple columns for Y-axis (e.g., multi-line plot).")
    color_by_column: Optional[str] = Field(None, description="Column to use for coloring/hue.")
    title: Optional[str] = Field(None, description="Suggested title for the plot.")
    xlabel: Optional[str] = Field(None, description="Suggested label for the X-axis.")
    ylabel: Optional[str] = Field(None, description="Suggested label for the Y-axis.") 
    bins: Optional[int] = Field(None, description="Number of bins (for histogram). Default is 'auto' if not provided or invalid.")
    
class AgentResponse(BaseModel):
    response_type: str
    content: Optional[str] = None
    plot_image_bytes: Optional[bytes] = None
    plot_config_json: Optional[str] = None 
    error: Optional[str] = None