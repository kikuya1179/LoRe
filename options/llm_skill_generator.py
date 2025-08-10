"""LLM-based skill/option generation for LoRe hierarchical RL."""

from __future__ import annotations

import json
import re
import uuid
from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np

from .option_framework import OptionSpec
from ..utils.code_out import CodeOut


class CrafterSkillLibrary:
    """Knowledge base of Crafter-specific skills and primitives."""
    
    def __init__(self):
        # Crafter action mapping (0-16)
        self.action_names = [
            "noop", "left", "right", "up", "down", 
            "do", "sleep", "place_stone", "place_table",
            "place_furnace", "place_plant", "make_wood_pickaxe",
            "make_stone_pickaxe", "make_iron_pickaxe", "make_wood_sword",
            "make_stone_sword", "make_iron_sword"
        ]
        
        # Common skill templates
        self.skill_templates = {
            "gather_resource": {
                "description": "Move to and collect a specific resource",
                "typical_actions": ["up", "down", "left", "right", "do"],
                "expected_duration": 8,
                "success_indicators": ["inventory_increase", "resource_collected"],
            },
            "craft_item": {
                "description": "Craft a specific item using available resources",
                "typical_actions": ["place_table", "place_furnace", "make_*"],
                "expected_duration": 5,
                "success_indicators": ["item_crafted", "inventory_change"],
            },
            "explore_area": {
                "description": "Systematically explore an area",
                "typical_actions": ["up", "down", "left", "right"],
                "expected_duration": 12,
                "success_indicators": ["area_coverage", "new_discoveries"],
            },
            "build_structure": {
                "description": "Build a structure or place items",
                "typical_actions": ["place_*", "left", "right", "up", "down"],
                "expected_duration": 6,
                "success_indicators": ["structure_built", "placement_success"],
            },
        }
        
        # Context-aware skill suggestions
        self.contextual_skills = {
            "low_health": ["find_bed", "sleep", "avoid_danger"],
            "night_time": ["sleep", "find_shelter", "light_area"],
            "low_resources": ["gather_wood", "gather_stone", "explore"],
            "crafting_needed": ["gather_materials", "place_table", "craft_tools"],
        }
    
    def get_action_id(self, action_name: str) -> Optional[int]:
        """Convert action name to ID."""
        try:
            return self.action_names.index(action_name.lower())
        except ValueError:
            # Try partial matching for make_* actions
            for i, name in enumerate(self.action_names):
                if action_name.lower() in name or name in action_name.lower():
                    return i
            return None
    
    def suggest_skills_for_context(self, context: Dict[str, Any]) -> List[str]:
        """Suggest appropriate skills based on game context."""
        suggestions = []
        
        # Analyze context and suggest skills
        health = context.get('health', 100)
        inventory = context.get('inventory', {})
        time_of_day = context.get('time', 'day')
        
        if health < 50:
            suggestions.extend(self.contextual_skills['low_health'])
        
        if time_of_day == 'night':
            suggestions.extend(self.contextual_skills['night_time'])
        
        if len(inventory) < 3:
            suggestions.extend(self.contextual_skills['low_resources'])
        
        if 'wood' in inventory and 'stone' in inventory:
            suggestions.extend(self.contextual_skills['crafting_needed'])
        
        return list(set(suggestions))  # Remove duplicates


class LLMSkillGenerator:
    """Generate executable skills/options using LLM guidance."""
    
    def __init__(
        self, 
        llm_adapter,
        skill_library: Optional[CrafterSkillLibrary] = None,
        max_skill_length: int = 10,
        confidence_threshold: float = 0.4,
    ):
        self.llm_adapter = llm_adapter
        self.skill_library = skill_library or CrafterSkillLibrary()
        self.max_skill_length = max_skill_length
        self.confidence_threshold = confidence_threshold
        
        # Generation statistics
        self.generation_stats = {
            'total_attempts': 0,
            'successful_generations': 0,
            'parsing_failures': 0,
            'validation_failures': 0,
            'avg_confidence': 0.0,
        }
    
    def generate_skills_for_context(
        self,
        obs: np.ndarray,
        context: Dict[str, Any],
        num_skills: int = 3,
    ) -> List[OptionSpec]:
        """Generate multiple skills appropriate for the current context."""
        
        skills = []
        self.generation_stats['total_attempts'] += 1
        
        try:
            # Get context-appropriate skill suggestions
            suggested_skill_types = self.skill_library.suggest_skills_for_context(context)
            
            # Build LLM prompt for skill generation
            prompt = self._build_skill_generation_prompt(
                context, suggested_skill_types, num_skills
            )
            
            # Call LLM
            if hasattr(self.llm_adapter, 'infer'):
                # Basic adapter
                llm_result = self.llm_adapter.infer(obs, num_actions=17)
                if isinstance(llm_result, dict) and 'prior_logits' in llm_result:
                    # Convert basic output to skills
                    skills = self._convert_basic_output_to_skills(llm_result, context)
                
            elif hasattr(self.llm_adapter, 'infer_batch'):
                # Enhanced adapter - try to get skill-specific output
                enhanced_context = context.copy()
                enhanced_context.update({
                    'request_type': 'skill_generation',
                    'num_skills': num_skills,
                    'skill_suggestions': suggested_skill_types,
                    'prompt': prompt,
                })
                
                results = self.llm_adapter.infer_batch([obs], [enhanced_context], 17)
                llm_result = results[0] if results and results[0] is not None else None
                
                if llm_result:
                    skills = self._parse_enhanced_skill_output(llm_result, context)
            
            # Filter and validate skills
            valid_skills = []
            for skill in skills:
                if self._validate_skill(skill):
                    valid_skills.append(skill)
                else:
                    self.generation_stats['validation_failures'] += 1
            
            if valid_skills:
                self.generation_stats['successful_generations'] += 1
                
                # Update confidence stats
                confidences = [skill.confidence for skill in valid_skills]
                self.generation_stats['avg_confidence'] = (
                    0.9 * self.generation_stats['avg_confidence'] + 
                    0.1 * np.mean(confidences)
                )
            
            return valid_skills
            
        except Exception as e:
            self.generation_stats['parsing_failures'] += 1
            print(f"Skill generation failed: {e}")
            return []
    
    def _build_skill_generation_prompt(
        self,
        context: Dict[str, Any],
        suggested_skills: List[str],
        num_skills: int,
    ) -> str:
        """Build prompt for LLM skill generation."""
        
        inventory = context.get('inventory', {})
        health = context.get('health', 100)
        step = context.get('step', 0)
        
        action_list = ", ".join([f"{i}:{name}" for i, name in enumerate(self.skill_library.action_names)])
        
        prompt = f"""You are a Crafter game AI assistant. Generate {num_skills} useful skills/macros for the current situation.

Current Context:
- Health: {health}
- Inventory: {inventory}
- Game step: {step}
- Suggested skills: {suggested_skills}

Available actions: {action_list}

Generate skills as JSON array with this format:
[
  {{
    "name": "gather_wood",
    "description": "Move right and collect wood",
    "actions": [2, 5, 5],
    "expected_duration": 3,
    "confidence": 0.8,
    "success_condition": "wood collected"
  }},
  ...
]

Each skill should:
1. Use action IDs (0-16) only
2. Be 3-8 actions long
3. Have clear purpose for current situation
4. Include realistic confidence (0.0-1.0)

Focus on: {', '.join(suggested_skills[:3])} if applicable.

Return ONLY the JSON array."""
        
        return prompt
    
    def _convert_basic_output_to_skills(
        self,
        llm_result: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[OptionSpec]:
        """Convert basic LLM adapter output to skills."""
        
        skills = []
        
        # Extract logits and convert to skill
        logits = llm_result.get('prior_logits', [])
        confidence = llm_result.get('confidence', [0.5])[0]
        
        if len(logits) == 17:
            # Use top actions as a skill sequence
            logits_tensor = torch.tensor(logits)
            top_k_actions = torch.topk(logits_tensor, k=min(5, len(logits))).indices.tolist()
            
            # Create a generic skill from top actions
            skill = OptionSpec(
                option_id=str(uuid.uuid4())[:8],
                name=f"llm_skill_basic",
                description="LLM suggested action sequence",
                primitive_actions=top_k_actions[:4],  # Limit to 4 actions
                expected_duration=len(top_k_actions[:4]),
                confidence=confidence,
                generation_context=context,
            )
            
            skills.append(skill)
        
        return skills
    
    def _parse_enhanced_skill_output(
        self,
        llm_result: CodeOut,
        context: Dict[str, Any],
    ) -> List[OptionSpec]:
        """Parse enhanced LLM adapter output for skills."""
        
        skills = []
        
        try:
            # Try to extract skills from notes or other fields
            notes = getattr(llm_result, 'notes', '')
            
            # Look for JSON in notes
            json_match = re.search(r'\\[.*\\]', notes, re.DOTALL)
            if json_match:
                try:
                    skill_data = json.loads(json_match.group())
                    skills = self._parse_skill_json(skill_data, context)
                except json.JSONDecodeError:
                    pass
            
            # Fallback: create skill from policy if available
            if not skills and hasattr(llm_result, 'policy') and llm_result.policy:
                if 'logits' in llm_result.policy:
                    logits = llm_result.policy['logits']
                    actions = torch.topk(torch.tensor(logits), k=4).indices.tolist()
                    
                    skill = OptionSpec(
                        option_id=str(uuid.uuid4())[:8],
                        name="llm_enhanced_skill",
                        description="Enhanced LLM skill",
                        primitive_actions=actions,
                        expected_duration=len(actions),
                        confidence=getattr(llm_result, 'confidence', 0.5),
                        generation_context=context,
                    )
                    skills.append(skill)
            
        except Exception as e:
            print(f"Failed to parse enhanced skill output: {e}")
        
        return skills
    
    def _parse_skill_json(
        self, 
        skill_data: List[Dict], 
        context: Dict[str, Any]
    ) -> List[OptionSpec]:
        """Parse JSON skill definitions."""
        
        skills = []
        
        for skill_dict in skill_data:
            try:
                # Validate required fields
                if not all(key in skill_dict for key in ['name', 'actions']):
                    continue
                
                # Convert action names to IDs if needed
                actions = skill_dict['actions']
                if isinstance(actions[0], str):
                    action_ids = []
                    for action_name in actions:
                        action_id = self.skill_library.get_action_id(action_name)
                        if action_id is not None:
                            action_ids.append(action_id)
                    actions = action_ids
                
                # Validate actions are in range
                if not all(0 <= a <= 16 for a in actions):
                    continue
                
                skill = OptionSpec(
                    option_id=str(uuid.uuid4())[:8],
                    name=skill_dict['name'],
                    description=skill_dict.get('description', ''),
                    primitive_actions=actions[:self.max_skill_length],
                    expected_duration=skill_dict.get('expected_duration', len(actions)),
                    confidence=float(skill_dict.get('confidence', 0.5)),
                    generation_context=context,
                    success_condition=skill_dict.get('success_condition'),
                    failure_condition=skill_dict.get('failure_condition'),
                )
                
                skills.append(skill)
                
            except Exception as e:
                print(f"Failed to parse skill: {e}")
                continue
        
        return skills
    
    def _validate_skill(self, skill: OptionSpec) -> bool:
        """Validate that a skill is executable and reasonable."""
        
        # Check basic requirements
        if not skill.primitive_actions:
            return False
        
        if len(skill.primitive_actions) > self.max_skill_length:
            return False
        
        if skill.confidence < self.confidence_threshold:
            return False
        
        # Check action validity
        if not all(0 <= a <= 16 for a in skill.primitive_actions):
            return False
        
        # Check for obviously bad patterns
        if len(set(skill.primitive_actions)) == 1 and skill.primitive_actions[0] == 0:
            # All no-op actions
            return False
        
        # Check for excessive movement without purpose
        movement_actions = [1, 2, 3, 4]  # left, right, up, down
        if (len(skill.primitive_actions) > 6 and 
            all(a in movement_actions for a in skill.primitive_actions)):
            return False
        
        return True
    
    def generate_fallback_skills(self, context: Dict[str, Any]) -> List[OptionSpec]:
        """Generate basic fallback skills when LLM generation fails."""
        
        fallback_skills = []
        
        # Basic exploration skill
        explore_skill = OptionSpec(
            option_id=str(uuid.uuid4())[:8],
            name="basic_explore",
            description="Basic exploration pattern",
            primitive_actions=[2, 2, 3, 3],  # right, right, up, up
            expected_duration=4,
            confidence=0.3,
            generation_context=context,
        )
        fallback_skills.append(explore_skill)
        
        # Basic gathering skill
        gather_skill = OptionSpec(
            option_id=str(uuid.uuid4())[:8],
            name="basic_gather",
            description="Basic gathering attempt",
            primitive_actions=[5, 1, 5, 2, 5],  # do, left, do, right, do
            expected_duration=5,
            confidence=0.3,
            generation_context=context,
        )
        fallback_skills.append(gather_skill)
        
        return fallback_skills
    
    def get_statistics(self) -> Dict[str, float]:
        """Get generation statistics."""
        stats = self.generation_stats.copy()
        
        if stats['total_attempts'] > 0:
            stats['success_rate'] = stats['successful_generations'] / stats['total_attempts']
            stats['parsing_failure_rate'] = stats['parsing_failures'] / stats['total_attempts']
            stats['validation_failure_rate'] = stats['validation_failures'] / stats['total_attempts']
        else:
            stats['success_rate'] = 0.0
            stats['parsing_failure_rate'] = 0.0
            stats['validation_failure_rate'] = 0.0
        
        return stats


class SkillEvaluator:
    """Evaluate and rank skills based on performance."""
    
    def __init__(self, evaluation_window: int = 20):
        self.evaluation_window = evaluation_window
        self.skill_performance_history: Dict[str, List[float]] = {}
    
    def record_skill_performance(
        self,
        skill_id: str,
        reward: float,
        success: bool,
        duration: int,
    ):
        """Record performance of a skill execution."""
        
        if skill_id not in self.skill_performance_history:
            self.skill_performance_history[skill_id] = []
        
        # Compute performance score
        performance_score = reward
        if success:
            performance_score += 0.1  # Bonus for success
        
        # Penalty for excessive duration
        expected_duration = 5  # Baseline
        if duration > expected_duration * 1.5:
            performance_score -= 0.05
        
        # Store performance
        self.skill_performance_history[skill_id].append(performance_score)
        
        # Limit history size
        if len(self.skill_performance_history[skill_id]) > self.evaluation_window:
            self.skill_performance_history[skill_id].pop(0)
    
    def get_skill_ranking(self, skill_ids: List[str]) -> List[Tuple[str, float]]:
        """Get skills ranked by average performance."""
        
        skill_scores = []
        
        for skill_id in skill_ids:
            if skill_id in self.skill_performance_history:
                history = self.skill_performance_history[skill_id]
                if history:
                    avg_score = np.mean(history[-self.evaluation_window:])
                    skill_scores.append((skill_id, avg_score))
                else:
                    skill_scores.append((skill_id, 0.0))
            else:
                skill_scores.append((skill_id, 0.0))
        
        # Sort by score descending
        skill_scores.sort(key=lambda x: x[1], reverse=True)
        
        return skill_scores
    
    def should_remove_skill(self, skill_id: str, min_score: float = -0.1) -> bool:
        """Check if skill should be removed due to poor performance."""
        
        if skill_id not in self.skill_performance_history:
            return False
        
        history = self.skill_performance_history[skill_id]
        if len(history) < self.evaluation_window // 2:
            return False  # Not enough data
        
        avg_score = np.mean(history[-self.evaluation_window:])
        return avg_score < min_score