"""
数据库 Pydantic v2 数据模型
 
每个模型严格对应一张 SQLite 表，同时按操作语义拆分为
Create / Update / Read 三个子模型，确保类型安全。
"""
 
from datetime import datetime
from enum import Enum
from typing import Optional
 
from pydantic import BaseModel, Field
 
 
# ═══════════════════════════════════════════════════════════
#  枚举
# ═══════════════════════════════════════════════════════════
 
 
class MarketType(str, Enum):
    """市场类型"""
    A = "a"
    HK = "hk"
    US = "us"
 
 
class AlertSeverity(str, Enum):
    """预警严重级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
 
 
class AlertEventType(str, Enum):
    """预警事件类型"""
    PRICE_SPIKE = "price_spike"
    VOLUME_SPIKE = "volume_spike"
    SECTOR_ROTATION = "sector_rotation"
    MARKET_CRASH = "market_crash"
    CUSTOM = "custom"
 
 
class RoleType(str, Enum):
    """对话角色"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
 
 
class SummaryType(str, Enum):
    """摘要类型"""
    CONVERSATION = "conversation"
    WEEKLY = "weekly"
    KEY_INSIGHT = "key_insight"


class ObservationStatus(str, Enum):
    """强势观察池状态"""
    ACTIVE = "active"
    DROPPED = "dropped"
    WATCHING = "watching"
 
 
# ═══════════════════════════════════════════════════════════
#  AppConfig
# ═══════════════════════════════════════════════════════════
 
 
class AppConfig(BaseModel):
    """运行时配置键值对（对应 app_config 表）"""
    key: str = Field(..., min_length=1, max_length=100, description="配置键")
    value: str = Field(..., description="配置值")
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(),
        description="最后更新时间",
    )
 
 
# ═══════════════════════════════════════════════════════════
#  Watchlist
# ═══════════════════════════════════════════════════════════
 
 
class WatchlistItemCreate(BaseModel):
    """添加自选股请求"""
    symbol: str = Field(..., min_length=1, max_length=20, description="股票代码")
    name: str = Field(..., min_length=1, max_length=50, description="股票名称")
    market: MarketType = Field(default=MarketType.A, description="所属市场")
    tags: str = Field(default="", max_length=200, description="标签（逗号分隔）")
    notes: str = Field(default="", max_length=500, description="用户备注")
 
 
class WatchlistItemUpdate(BaseModel):
    """更新自选股请求（所有字段可选）"""
    name: Optional[str] = Field(default=None, max_length=50, description="股票名称")
    market: Optional[MarketType] = Field(default=None, description="所属市场")
    tags: Optional[str] = Field(default=None, max_length=200, description="标签")
    notes: Optional[str] = Field(default=None, max_length=500, description="用户备注")
 
 
class WatchlistItem(WatchlistItemCreate):
    """自选股完整记录"""
    id: int = Field(..., description="主键 ID")
    added_at: datetime = Field(..., description="添加时间")
 
    updated_at: Optional[datetime] = Field(default=None, description="最后更新时间")

    model_config = {"from_attributes": True}
 
 
# ═══════════════════════════════════════════════════════════
#  Conversations
# ═══════════════════════════════════════════════════════════
 
 
class ConversationMessageCreate(BaseModel):
    """写入对话记录请求"""
    session_id: str = Field(..., max_length=100, description="会话 ID（用户 open_id）")
    role: RoleType = Field(..., description="角色")
    content: str = Field(..., description="消息内容")
    metadata: str = Field(default="{}", description="扩展元数据（JSON 字符串）")
    conversation_id: str = Field(default="", max_length=50, description="对话轮次标识")
 
 
class ConversationMessage(ConversationMessageCreate):
    """对话记录完整记录"""
    id: int = Field(..., description="主键 ID")
    is_summarized: int = Field(default=0, ge=0, le=1, description="是否已被摘要折叠（0/1）")
    created_at: datetime = Field(..., description="创建时间")
 
    model_config = {"from_attributes": True}
 
 
# ═══════════════════════════════════════════════════════════
#  Conversation Summaries
# ═══════════════════════════════════════════════════════════
 
 
class ConversationSummaryCreate(BaseModel):
    """创建摘要请求"""
    session_id: str = Field(..., max_length=100, description="会话 ID")
    summary: str = Field(..., description="摘要内容")
    summary_type: SummaryType = Field(
        default=SummaryType.CONVERSATION,
        description="摘要类型",
    )
    token_count: int = Field(default=0, ge=0, description="Token 数量")
 
 
class ConversationSummary(ConversationSummaryCreate):
    """摘要完整记录"""
    id: int = Field(..., description="主键 ID")
    created_at: datetime = Field(..., description="创建时间")
 
    model_config = {"from_attributes": True}
 
 
# ═══════════════════════════════════════════════════════════
#  Alert Events
# ═══════════════════════════════════════════════════════════
 
 
class AlertEventCreate(BaseModel):
    """创建预警事件请求"""
    event_id: str = Field(..., max_length=100, description="事件唯一标识")
    alert_type: AlertEventType = Field(..., description="预警类型")
    title: str = Field(..., max_length=200, description="预警标题")
    content: str = Field(..., description="预警详情")
    severity: AlertSeverity = Field(
        default=AlertSeverity.INFO,
        description="严重级别",
    )
    related_code: str = Field(default="", max_length=20, description="关联股票代码")
    related_sector: str = Field(default="", max_length=50, description="关联板块名称")
    strength: float = Field(default=0.0, ge=0.0, le=100.0, description="当前强度值")
 
 
class AlertEventUpdate(BaseModel):
    """更新预警事件（去重/提级逻辑使用）"""
    strength: float = Field(..., ge=0.0, le=100.0, description="当前强度值")
    severity: AlertSeverity = Field(default=AlertSeverity.INFO, description="严重级别")
    sent_count: int = Field(default=0, ge=0, description="累计推送次数")
 
 
class AlertEvent(AlertEventCreate):
    """预警事件完整记录"""
    id: int = Field(..., description="主键 ID")
    peak_strength: float = Field(default=0.0, ge=0.0, description="历史峰值强度")
    first_seen: Optional[datetime] = Field(default=None, description="首次发现时间")
    last_sent: Optional[datetime] = Field(default=None, description="最近推送时间")
    sent_count: int = Field(default=0, ge=0, description="累计推送次数")
    resolved: int = Field(default=0, ge=0, le=1, description="是否已解除（0/1）")
    created_at: datetime = Field(..., description="创建时间")
 
    model_config = {"from_attributes": True}


# ═══════════════════════════════════════════════════════════
#  Observation Pool
# ═══════════════════════════════════════════════════════════


class ObservationPoolEntry(BaseModel):
    """强势观察池记录"""
    symbol: str = Field(..., min_length=1, max_length=20, description="股票代码")
    name: str = Field(..., max_length=50, description="股票名称")
    industry: str = Field(default="", max_length=50, description="所属行业")
    first_seen: str = Field(..., description="首次进入观察池日期")
    last_seen: str = Field(..., description="最近进入 Top3 日期")
    consecutive_days: int = Field(default=1, ge=0, description="连续上榜天数")
    highest_score: float = Field(default=0.0, ge=0.0, description="历史最高评分")
    latest_score: float = Field(default=0.0, ge=0.0, description="最新评分")
    latest_rank: int = Field(default=0, ge=0, description="最新排名")
    latest_reason: str = Field(default="", description="最新强势原因")
    status: ObservationStatus = Field(
        default=ObservationStatus.ACTIVE,
        description="观察池状态",
    )

    model_config = {"from_attributes": True}
 
 
# ═══════════════════════════════════════════════════════════
#  扩展字段基类（可嵌入 metadata 使用）
# ═══════════════════════════════════════════════════════════
 
 
class ExtraMetadata(BaseModel):
    """预留扩展元数据结构"""
    source: str = Field(default="", description="数据来源")
    tags: list[str] = Field(default_factory=list, description="标签列表")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="置信度")
    custom: dict[str, object] = Field(default_factory=dict, description="自定义扩展")
