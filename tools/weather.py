"""
天气工具（Mock 实现）
"""
from typing import Any, Dict
from .base import BaseTool, ToolParameter


class WeatherTool(BaseTool):
    """天气查询工具 - Mock 实现"""
    
    def __init__(self):
        super().__init__()
        self.name = "weather"
        self.description = "查询指定城市的天气信息"
        self.parameters = [
            ToolParameter(
                name="city",
                type="string",
                description="城市名称，例如: '北京'、'上海'、'New York'"
            )
        ]
        
        # Mock 天气数据
        self._mock_weather = {
            "北京": {"temp": 22, "condition": "晴", "humidity": 45, "wind": "北风3级"},
            "上海": {"temp": 26, "condition": "多云", "humidity": 72, "wind": "东风2级"},
            "广州": {"temp": 30, "condition": "阵雨", "humidity": 85, "wind": "南风2级"},
            "深圳": {"temp": 29, "condition": "阴", "humidity": 80, "wind": "东南风3级"},
            "杭州": {"temp": 25, "condition": "晴转多云", "humidity": 65, "wind": "东风1级"},
            "new york": {"temp": 18, "condition": "Rainy", "humidity": 78, "wind": "NW 5mph"},
            "london": {"temp": 15, "condition": "Cloudy", "humidity": 82, "wind": "SW 8mph"},
        }
    
    def execute(self, city: str) -> Dict[str, Any]:
        """查询天气"""
        city_lower = city.lower()
        
        # 查找匹配的城市
        weather_data = None
        matched_city = None
        
        for key, data in self._mock_weather.items():
            if key.lower() in city_lower or city_lower in key.lower():
                weather_data = data
                matched_city = key
                break
        
        if not weather_data:
            return {
                "success": False,
                "error": f"未找到城市 '{city}' 的天气数据",
                "available_cities": list(self._mock_weather.keys())
            }
        
        return {
            "success": True,
            "city": matched_city,
            "temperature": f"{weather_data['temp']}°C",
            "condition": weather_data["condition"],
            "humidity": f"{weather_data['humidity']}%",
            "wind": weather_data["wind"],
            "suggestion": self._get_suggestion(weather_data)
        }
    
    def _get_suggestion(self, weather_data: Dict) -> str:
        """根据天气给出建议"""
        temp = weather_data["temp"]
        condition = weather_data["condition"]
        
        suggestions = []
        
        if temp > 28:
            suggestions.append("天气炎热，注意防暑降温")
        elif temp < 10:
            suggestions.append("天气寒冷，注意保暖")
        
        if "雨" in condition or "rain" in condition.lower():
            suggestions.append("记得带伞")
        
        if weather_data["humidity"] > 75:
            suggestions.append("湿度较高，注意防潮")
        
        return "；".join(suggestions) if suggestions else "天气适宜，适合外出"
