"""
Qwen model wrapper for TicTacToe self-play
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  PEFT未安装，无法加载LoRA模型")


class QwenWrapper:
    """Wrapper for Qwen model to generate TicTacToe moves with CoT reasoning"""
    
    def __init__(self, model_path: str = None, device: str = "auto", use_lora: bool = False, base_model_path: str = None):
        self.model_path = model_path or self._get_default_model_path()
        self.device = device
        self.use_lora = use_lora
        self.base_model_path = base_model_path or self._get_default_base_model_path()
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
    def _get_default_model_path(self) -> str:
        """Get default model path relative to project root"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        return os.path.join(project_root, "qwen")
    
    def _get_default_base_model_path(self) -> str:
        """Get default base model path"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        return os.path.join(project_root, "qwen")
    
    def load_model(self):
        """Load the Qwen model and tokenizer"""
        try:
            if self.use_lora:
                if not PEFT_AVAILABLE:
                    raise ImportError("PEFT not available for LoRA loading")
                print(f"Loading LoRA model from: {self.model_path}")
                print(f"Base model: {self.base_model_path}")
                
                # Load tokenizer from base model
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_path,
                    trust_remote_code=True
                )
                
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map=self.device if self.device != "auto" else "auto",
                    trust_remote_code=True
                )
                
                # Load LoRA adapter
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
                
            else:
                print(f"Loading base Qwen model from: {self.model_path}")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map=self.device if self.device != "auto" else "auto",
                    trust_remote_code=True
                )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.is_loaded = True
            print(f"Model loaded successfully on device: {self.model.device}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Falling back to rule-based strategy")
            self.is_loaded = False
    
    def generate_move_with_cot(self, observation: str, player_mark: str = "X") -> Tuple[str, str]:
        """
        Generate a move with Chain of Thought reasoning
        
        Args:
            observation: Current game state description
            player_mark: "X" or "O"
            
        Returns:
            (cot_reasoning, action) tuple
        """
        
        if not self.is_loaded:
            return self._fallback_strategy(observation, player_mark)
        
        try:
            # Create prompt for CoT reasoning
            prompt = self._create_cot_prompt(observation, player_mark)
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            # Parse CoT and action
            cot, action = self._parse_response(response, observation)
            return cot, action
            
        except Exception as e:
            print(f"Error in model generation: {e}")
            return self._fallback_strategy(observation, player_mark)
    
    def _create_cot_prompt(self, observation: str, player_mark: str) -> str:
        """Create a prompt for Chain of Thought reasoning"""
        prompt = f"""你是一个井字棋专家。请分析当前棋盘状态，详细思考最优落子位置。

当前游戏状态：
{observation}

你的角色是{player_mark}。请按以下格式思考和回答：

思考过程：
1. 分析当前棋盘：描述当前局面
2. 寻找获胜机会：检查是否能直接获胜
3. 防守策略：检查对手是否即将获胜，需要阻止
4. 战略位置：评估中心、角落、边缘位置的价值
5. 最终决策：选择最优位置

答案格式：答案: [位置编号]

开始分析："""
        
        return prompt
    
    def _parse_response(self, response: str, observation: str) -> Tuple[str, str]:
        """Parse model response to extract CoT and action"""
        import re
        
        # Try to find action in format "答案: [数字]" or similar
        action_patterns = [
            r'答案\s*:\s*\[(\d+)\]',
            r'答案\s*:\s*(\d+)',
            r'选择\s*:\s*\[(\d+)\]',
            r'选择\s*:\s*(\d+)',
            r'位置\s*:\s*\[(\d+)\]',
            r'位置\s*:\s*(\d+)',
            r'\[(\d+)\]'
        ]
        
        action = None
        for pattern in action_patterns:
            match = re.search(pattern, response)
            if match:
                action = match.group(1)
                break
        
        # Extract CoT (everything before the final answer)
        if "答案:" in response:
            cot = response.split("答案:")[0].strip()
        else:
            cot = response.strip()
        
        # Validate action
        if action is None or not action.isdigit() or not (0 <= int(action) <= 8):
            # Fall back to extracting available moves and choosing one
            available_moves = self._extract_available_moves(observation)
            if available_moves:
                action = str(available_moves[0])  # Choose first available
            else:
                action = "4"  # Default to center
        
        return cot, action
    
    def _extract_available_moves(self, observation: str) -> list:
        """Extract available moves from observation"""
        import re
        if "Available Moves" in observation:
            moves_str = observation.split("Available Moves:")[1]
            moves = re.findall(r'\[(\d+)\]', moves_str)
            return [int(m) for m in moves]
        return list(range(9))  # Fallback
    
    def _fallback_strategy(self, observation: str, player_mark: str) -> Tuple[str, str]:
        """Fallback strategy when model is not available"""
        available_moves = self._extract_available_moves(observation)
        
        cot = f"""
思考过程（规则策略）：
1. 当前是{player_mark}，可用位置：{available_moves}
2. 优先选择中心位置(4)
3. 其次选择角落位置(0,2,6,8)
4. 最后选择边缘位置
"""
        
        # Simple priority strategy
        priority_moves = [4, 0, 2, 6, 8, 1, 3, 5, 7]
        for move in priority_moves:
            if move in available_moves:
                return cot, str(move)
        
        return cot, str(available_moves[0]) if available_moves else "0"

    def predict(self, observation):
        """Legacy method for compatibility"""
        cot, action = self.generate_move_with_cot(observation)
        return int(action)

    def act(self, observation):
        """Generate an action based on the current game state"""
        prediction = self.predict(observation)
        return prediction