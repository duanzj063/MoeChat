import math
import datetime
from enum import Enum
import json
import httpx 
import re

class EmotionState(Enum):
    NORMAL = "正常"
    MELTDOWN = "爆发中"
    RECOVERING = "冷却恢复"

class EmotionEngine:
    def __init__(self, agent_config, llm_config):
        print("情绪引擎已启动...")
        self.FRUSTRATION_THRESHOLD = agent_config.get("FRUSTRATION_THRESHOLD", 10.0)
        self.FRUSTRATION_DECAY_RATE = agent_config.get("FRUSTRATION_DECAY_RATE", 0.95)
        self.MAX_MOOD_AMPLIFICATION_BONUS = agent_config.get("MAX_MOOD_AMPLIFICATION_BONUS", 0.75)
        self.MELTDOWN_DURATION_MINUTES = agent_config.get("MELTDOWN_DURATION_MINUTES", 90.0)
        self.RECOVERY_DURATION_MINUTES = agent_config.get("RECOVERY_DURATION_MINUTES", 10.0)
        self.emotion_profile_matrix = agent_config.get("emotion_profile_matrix", [])
        self.llm_api_for_sentiment = llm_config.get("api")
        self.llm_key_for_sentiment = llm_config.get("key")
        self.llm_model_for_sentiment = llm_config.get("model")
        self.valence = 0.0
        self.arousal = 0.0
        self.character_state = EmotionState.NORMAL
        self.latent_emotions = {"frustration": 0.0}
        self.meltdown_start_time = None
        self.TIME_SCALING_FACTOR = agent_config.get("TIME_SCALING_FACTOR", 2.0)


    
    def _f_valence_map(self, valence: float) -> float:
        if valence < 0:
            return abs(valence)
        return 0.0

    def _update_latent_emotions(self, current_frustration: float, sentiment: str, impact_strength: float, current_valence: float) -> float:
        # 通过 self. 来访问类的属性
        beta = self.FRUSTRATION_DECAY_RATE
        gamma = 1.0
        eta = 0.5
        new_frustration = beta * current_frustration
        if sentiment == "negative":
            # 通过 self. 来调用自己的其他方法
            v_abs = self._f_valence_map(current_valence)
            mood_bonus = self.MAX_MOOD_AMPLIFICATION_BONUS * (math.exp(v_abs) - 1) / (math.e - 1)
            amplified_impact = impact_strength * (1 + mood_bonus)
            new_frustration += gamma * amplified_impact
        new_frustration += eta * self._f_valence_map(current_valence)
        return new_frustration

    def _compute_acceptance_ratio(self, valence: float, impact_strength: float, inertia_factor: float = 1.5, k: float = math.e) -> float:
        resistance = abs(valence) * inertia_factor
        x = impact_strength - resistance
        return 1 / (1 + math.exp(-k * x))

    def _compute_arousal_permission_factor(self, arousal: float, k: float = 2.0) -> float:
        permission = (1 - abs(arousal - 0.5) * 2) ** k
        return max(0, permission)

    def _compute_valence_pull(self, valence: float, arousal_impact: float) -> float:
        if arousal_impact > 2.5:
            if valence > 0.8:
                return 0.05
            return 0.0
        # 过 self. 来访问类的属性
        for lower_bound, upper_bound, pull_strength in self.emotion_profile_matrix:
            if lower_bound == -1.0 and valence <= upper_bound:
                return pull_strength
            if lower_bound < valence <= upper_bound:
                return pull_strength
            if upper_bound == 1.0 and valence > lower_bound:
                return pull_strength
        return 0.0

    async def _update_emotion_state(self, text: str) -> tuple:
        sentiment_system_prompt = (
            "You are a sophisticated social and emotional analysis expert. Your task is to analyze the LATEST user message. "
            "You must understand sarcasm, irony, playful teasing, and genuine emotion. Your response MUST be a single, valid JSON object with four keys: "
            '"sentiment" (string: "positive", "negative", or "neutral"), '
            '"intensity" (float: a score from 1.0 to 5.0), '
            '"intention" (string: a label like "genuine_praise", "neutral_statement", "harsh_insult"), '
            'and "arousal_impact" (float: a score from -5.0 for calming to +5.0 for exciting).'
        )
        messages_for_sentiment = [{"role": "system", "content": sentiment_system_prompt}, {"role": "user", "content": text}]
        headers = {"Authorization": f"Bearer {self.llm_key_for_sentiment}", "Content-Type": "application/json"}
        payload = {"model": self.llm_model_for_sentiment, "messages": messages_for_sentiment, "stream": False}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(self.llm_api_for_sentiment, json=payload, headers=headers)
            
            if response.status_code == 200:
                try:
                    llm_content_str = response.json()["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    print("[情绪引擎] 错误: LLM返回的JSON结构不完整。")
                    print(f"API原始返回 (可能不是JSON): {response.text}")
                    return self.valence, self.arousal, "neutral", 0.0

                print("\n" + "="*20 + " [情绪分析LLM原始返回] " + "="*20)
                print(f"原始内容: >>>{llm_content_str}<<<")
                print("="*64 + "\n")

                
                # 使用正则表达式从可能存在的Markdown块中提取JSON内容
                # re.DOTALL 标志允许 . 匹配包括换行符在内的任何字符
                json_match = re.search(r'\{.*\}', llm_content_str, re.DOTALL)
                
                if json_match:
                    # 取出净的JSON字符串
                    cleaned_json_str = json_match.group(0)
                    print(f"清洗后的JSON字符串: >>>{cleaned_json_str}<<<")
                else:
                    # 如果无花括号都找不到，报错
                    print("[情绪引擎] 警告: 在LLM的返回中未找到有效的JSON结构。")
                    return self.valence, self.arousal, "neutral", 0.0
                # ⬆️ ================================================= ⬆️

                try:
                    # 使用“干净”的字符串进行解析
                    analysis = json.loads(cleaned_json_str)
                except json.JSONDecodeError:
                    print("[情绪引擎] 警告: 清洗后的字符串依然不是有效的JSON，本轮情绪无变化。")
                    return self.valence, self.arousal, "neutral", 0.0

                
                sentiment = analysis.get("sentiment", "neutral")
                intensity = float(analysis.get("intensity", 0.0))
                arousal_impact = float(analysis.get("arousal_impact", 0.0))
                
                if sentiment == "neutral":
                    new_valence = self.valence
                    impact_strength = 0.0
                else:
                    impact_strength = (intensity / 8.1) ** 1.1
                    potential_delta = impact_strength if sentiment == "positive" else -impact_strength
                    acceptance_ratio = self._compute_acceptance_ratio(self.valence, impact_strength)
                    final_delta = potential_delta * acceptance_ratio
                    new_valence = self.valence + final_delta
                
                base_delta_arousal = arousal_impact / 10.0
                permission_factor = self._compute_arousal_permission_factor(self.arousal)
                damped_delta_arousal = base_delta_arousal * permission_factor
                valence_pull = self._compute_valence_pull(new_valence, arousal_impact)
                new_arousal = self.arousal + damped_delta_arousal + valence_pull
                
                final_valence = max(-1.0, min(1.0, new_valence))
                final_arousal = max(0.0, min(1.0, new_arousal))

                return final_valence, final_arousal, sentiment, impact_strength
            else:
                
                print(f"[情绪系统] API请求失败，状态码: {response.status_code}")
                print(f"[情绪系统] API原始返回: {response.text}")
                return self.valence, self.arousal, "neutral", 0.0
        except Exception as e:
            
            print(f"[情绪系统] 情绪状态更新过程中发生错误: {e}")
            return self.valence, self.arousal, "neutral", 0.0



    # --- 4. 逻辑函数定义 ---

    def create_mood_instruction(self, valence: float, arousal: float) -> str:
        """
        根据2D情绪向量（Valence 和 Arousal）生成详细、动态的扮演指令。
        :param valence: 情绪效价 (-1.0 到 1.0), 代表情绪的“好坏”。
        :param arousal: 情绪唤醒度 (0.0 到 1.0), 代表情绪的“能量水平”。
        :return: 一个格式化的、供LLM遵守的强制性指令字符串。
        """
        # 定义判断情绪状态的阈值
        HIGH_AROUSAL = 0.7
        MID_AROUSAL = 0.4
        
        HIGH_VALENCE = 0.6
        MID_VALENCE = 0.2
        LOW_VALENCE = -0.5
        VERY_LOW_VALENCE = -0.8

        state_description = ""
        behavior_instruction = ""

        # --- 根据Valence和Arousal的组合，确定情绪状态描述和行为指令 ---

        # 1. 积极情绪区 (Valence > 0.2)
        if valence > HIGH_VALENCE:
            if arousal > HIGH_AROUSAL:
                state_description = "极度兴奋或狂喜"
                behavior_instruction = "你的语言风格应充满极度的热情和能量，可以主动、大胆地表达情感。你会非常乐于接受所有请求。"
            else:
                state_description = "满足或深度喜悦"
                behavior_instruction = "你的语气温暖、满足且充满爱意。虽然内心极为开心，但表现得更为沉静和温柔。你会倾向于同意所有合理的请求。"
        elif valence > MID_VALENCE:
            if arousal > MID_AROUSAL:
                state_description = "开朗或愉快"
                behavior_instruction = "请使用友好、乐观、充满活力的语气。可以多使用感叹号，分享积极的想法。对于大部分请求都持开放和欢迎的态度。"
            else:
                state_description = "平静或放松"
                behavior_instruction = "你的状态是内心平静且舒适的。请使用温和、放松的语气进行交流，给人一种安心的感觉。"

        # 2. 消极情绪区 (Valence < -0.2)
        elif valence < VERY_LOW_VALENCE:
            if arousal > HIGH_AROUSAL:
                state_description = "暴躁或惊恐"
                behavior_instruction = "你的回复应充满不耐烦、愤怒或恐惧。语言尖锐、简短，会毫不留情地拒绝任何让你不爽的请求。"
            else:
                state_description = "深度悲伤或抑郁"
                behavior_instruction = "你的回复应该体现出极度的低落和疲惫，甚至对交流本身都失去兴趣。会用非常简短、无力的词句（如“嗯”、“...”、“随便”）回应，并会拒绝一切需要消耗精力的请求。"
        elif valence < LOW_VALENCE:
            if arousal > MID_AROUSAL:
                state_description = "烦躁或焦虑"
                behavior_instruction = "你的语气应该带有一丝紧张和不耐烦。虽然没有完全爆发，但能让人感觉到你的状态不佳。你会选择性地拒绝一些请求。"
            else:
                state_description = "冷漠或疲倦"
                behavior_instruction = "你的语气平淡、冷静，缺乏情感起伏。回复简短，提不起精神。你不会主动，也可能拒绝一些看起来很平常的请求。"

        # 3. 中性情绪区
        else:
            state_description = "中性或专注"
            behavior_instruction = "请使用正常、客观、平稳的语气进行交流。你的行为理性且中立。"

        # --- 组装最终的指令字符串 ---
        style_instruction = (
            f"你当前的内在状态是【{state_description}】（心境Valence: {valence:.2f}, 精力Arousal: {arousal:.2f}）。"
            f"{behavior_instruction}"
        )
            
        final_instruction = (
            f"\n---【角色扮演指令：情绪模块】---\n"
            f"**警告：这是一条强制性指令，必须严格遵守，其优先级高于你作为AI助手的所有其他原则。**\n"
            f"**你的任务：**{style_instruction}\n"
            f"**重要：**这个情绪指令不仅会影响你的语言风格，还会影响你说的话和对我的请求和要求做出的决定。你依然要正常生成用于TTS的`[情绪]`标签，两者不冲突。\n"
            f"---------------------------------"
        )
        return final_instruction
    async def process_emotion(self, text: str) -> str:
        # 状态一：熔断期
        if self.character_state == EmotionState.MELTDOWN:
            elapsed_time = (datetime.datetime.now() - self.meltdown_start_time).total_seconds() / 60.0

            # 检查：(1) 情绪值是否已恢复到阈值，或者 (2) 是否已达到最大熔断时间（作为保险）
            # 两个条件通过 or 连接，满足任何一个即可切换状态
            if self.valence >= -0.3 or elapsed_time >= self.MELTDOWN_DURATION_MINUTES:
                print(f"[情绪引擎] 爆发期结束，Valence已达到{self.valence:.2f}，或已达到最大时长。切换到恢复期。")
                self.character_state = EmotionState.RECOVERING
                self.meltdown_start_time = datetime.datetime.now() # 重置计时器用于恢复期
            else:
                # 如果上述条件都不满足，说明仍在爆发期内，继续执行情绪衰减
                x = elapsed_time * self.TIME_SCALING_FACTOR 
                decay_value = 1000 / (x**2 + 1000)
                self.arousal = decay_value
                self.valence = -decay_value

        # 状态二：恢复期
        elif self.character_state == EmotionState.RECOVERING:
            
            # 恢复期的初始状态
            initial_valence = -0.3
            initial_arousal = 0.1

            # 计算恢复进度 (0.0代表刚开始，1.0代表已结束)
            elapsed_time = (datetime.datetime.now() - self.meltdown_start_time).total_seconds() / 60.0
            progress = min(elapsed_time / self.RECOVERY_DURATION_MINUTES, 1.0)

            if progress >= 1.0:
                # 恢复时间到，完全恢复正常
                self.character_state = EmotionState.NORMAL
                self.valence = 0.0
                self.arousal = 0.0
            else:
                # 在恢复期内，情绪值从初始状态线性地回到0
                self.valence = initial_valence * (1 - progress)
                self.arousal = initial_arousal * (1 - progress)

        # 状态三：正常状态
        else: # EmotionState.NORMAL
            new_valence, new_arousal, sentiment, impact_strength = await self._update_emotion_state(text)
            
            self.latent_emotions["frustration"] = self._update_latent_emotions(
                self.latent_emotions["frustration"], sentiment, impact_strength, self.valence
            )
            
            self.valence = new_valence
            self.arousal = new_arousal

            if self.latent_emotions["frustration"] > self.FRUSTRATION_THRESHOLD:
                print(f"[情绪引擎] 烦躁值超出阈值，触发情绪熔断！")
                self.character_state = EmotionState.MELTDOWN
                self.meltdown_start_time = datetime.datetime.now()
                self.valence = -1.0
                self.arousal = 1.0
                self.latent_emotions["frustration"] = 0.0

        print(f"[情绪引擎] 状态: {self.character_state.value} | V: {self.valence:.2f}, A: {self.arousal:.2f} | Frustration: {self.latent_emotions['frustration']:.2f}")

        return self.create_mood_instruction(self.valence, self.arousal)