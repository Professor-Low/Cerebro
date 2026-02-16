"""
Ultimate AI Memory System - Comprehensive Brain for Personal AI
Captures EVERYTHING: facts, preferences, file paths, decisions, actions, code, and more
"""
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import robust extractor (Production-ready extraction)
try:
    from robust_extractor import RobustExtractor
    ROBUST_EXTRACTION_ENABLED = True
except ImportError:
    ROBUST_EXTRACTION_ENABLED = False
    print("Warning: robust_extractor not found. Using legacy extraction.")

# Import user personal profile extraction
try:
    from personal_profile_extractor import PersonalProfileExtractor, update_user_profile
    USER_EXTRACTION_ENABLED = True
except ImportError:
    USER_EXTRACTION_ENABLED = False
    print("Warning: personal_profile_extractor not found. User profile updates disabled.")

# Import corrections detection (Agent 2 integration)
try:
    from corrections_integration import process_conversation_for_corrections
    CORRECTIONS_ENABLED = True
except ImportError:
    CORRECTIONS_ENABLED = False
    print("Warning: Corrections integration not available")

# Import project tracking (Agent 3 integration)
try:
    from project_integration import process_conversation_for_projects
    PROJECT_TRACKING_ENABLED = True
except ImportError:
    PROJECT_TRACKING_ENABLED = False
    print("Warning: Project tracking not available")

# Import temporal classification and preferences (Agent 8 integration)
try:
    from preference_manager import PreferenceManager
    from temporal_classifier import TemporalClassifier
    TEMPORAL_PREFERENCES_ENABLED = True
except ImportError:
    TEMPORAL_PREFERENCES_ENABLED = False
    print("Warning: Temporal classification and preferences not available")

# Import confidence tracking (Phase 3 - Brain Evolution)
try:
    from confidence_tracker import ConfidenceSource, ConfidenceTracker
    CONFIDENCE_TRACKING_ENABLED = True
except ImportError:
    CONFIDENCE_TRACKING_ENABLED = False
    print("Warning: Confidence tracking not available")

# Import privacy filter (Phase 5 - Brain Evolution)
try:
    from privacy_filter import PrivacyFilter, SensitivityLevel
    PRIVACY_FILTER_ENABLED = True
except ImportError:
    PRIVACY_FILTER_ENABLED = False
    print("Warning: Privacy filter not available")

# Import provenance tracking (Phase 6 - Brain Evolution)
try:
    from provenance_tracker import LearnedType, ProvenanceTracker, detect_learned_type
    PROVENANCE_TRACKING_ENABLED = True
except ImportError:
    PROVENANCE_TRACKING_ENABLED = False
    print("Warning: Provenance tracking not available")


class UltimateMemoryService:
    """
    Comprehensive AI memory system that captures and structures ALL information
    from conversations to build a queryable knowledge base about the user's life.

    Supports incremental saves - continue a conversation and only save new messages.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:

            from config import AI_MEMORY_BASE

            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self.conversations_path = self.base_path / "conversations"
        self.facts_path = self.base_path / "facts"
        self.metadata_path = self.base_path / "metadata"
        self.entities_path = self.base_path / "entities"
        self.timeline_path = self.base_path / "timeline"
        self.session_states_path = self.base_path / "session_states"

        # LAZY directory creation - don't block in constructor
        self._directories_created = False

        # Initialize user profile extractor if available
        if USER_EXTRACTION_ENABLED:
            self.user_extractor = PersonalProfileExtractor()
        else:
            self.user_extractor = None

        # Initialize temporal classifier and preference manager if available
        if TEMPORAL_PREFERENCES_ENABLED:
            self.temporal_classifier = TemporalClassifier()
            self.preference_manager = PreferenceManager(base_path=str(self.base_path))
        else:
            self.temporal_classifier = None
            self.preference_manager = None

        # Initialize confidence tracker if available (Phase 3 - Brain Evolution)
        if CONFIDENCE_TRACKING_ENABLED:
            self.confidence_tracker = ConfidenceTracker(base_path=str(self.base_path))
        else:
            self.confidence_tracker = None

        # Initialize privacy filter if available (Phase 5 - Brain Evolution)
        if PRIVACY_FILTER_ENABLED:
            self.privacy_filter = PrivacyFilter(
                redact_secrets=True,
                min_confidence=0.70,
                tag_sensitive=True
            )
        else:
            self.privacy_filter = None

        # Initialize provenance tracker if available (Phase 6 - Brain Evolution)
        if PROVENANCE_TRACKING_ENABLED:
            self.provenance_tracker = ProvenanceTracker(base_path=str(self.base_path))
        else:
            self.provenance_tracker = None

    def _ensure_directories(self) -> bool:
        """Lazily create directories only when needed"""
        if self._directories_created:
            return True

        try:
            for path in [self.conversations_path, self.facts_path, self.metadata_path,
                         self.entities_path, self.timeline_path, self.session_states_path]:
                path.mkdir(parents=True, exist_ok=True)
            self._directories_created = True
            return True
        except Exception as e:
            print(f"WARNING: Failed to create directories: {e}")
            return False

    def save_conversation(self,
                         messages: List[Dict[str, str]],
                         session_id: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None,
                         incremental: bool = False) -> str:
        """
        Save a conversation with COMPREHENSIVE extraction of:
        - Facts and learnings
        - File paths and system details
        - User preferences and goals
        - Actions taken and decisions made
        - Problems solved
        - Code snippets
        - Entities (people, tools, technologies)
        - Timeline of events

        Args:
            messages: List of message dicts with 'role' and 'content'
            session_id: Optional session identifier for linking conversations
            metadata: Optional additional metadata
            incremental: If True, only saves NEW messages since last save for this session_id
                        Requires session_id to be set. Avoids duplicate embeddings.

        Returns:
            conversation_id: The ID of the saved conversation segment
        """

        # Handle incremental saves
        if incremental:
            if not session_id:
                raise ValueError("incremental=True requires session_id to be set")

            # Get session state to find what was already saved
            session_state = self._get_session_state(session_id)
            last_saved_count = session_state.get("message_count", 0)

            if last_saved_count >= len(messages):
                # Nothing new to save
                return session_state.get("last_conversation_id", session_id)

            # Only process new messages
            new_messages = messages[last_saved_count:]
            parent_conv_id = session_state.get("last_conversation_id")

            # Generate segment ID (session_id + segment number)
            segment_num = session_state.get("segment_count", 0) + 1
            conv_id = f"{session_id}_seg{segment_num}"

            # Update metadata to link segments
            metadata = metadata or {}
            metadata["is_incremental"] = True
            metadata["parent_session"] = session_id
            metadata["segment_number"] = segment_num
            if parent_conv_id:
                metadata["parent_conversation_id"] = parent_conv_id
            metadata["message_range"] = {
                "start": last_saved_count,
                "end": len(messages)
            }

            # Use new messages for processing
            messages_to_process = new_messages
        else:
            messages_to_process = messages
            # Generate conversation ID
            if session_id:
                conv_id = session_id
            else:
                content_hash = hashlib.sha256(
                    json.dumps(messages).encode()
                ).hexdigest()[:16]
                conv_id = content_hash

        # Combine all message content (only process new messages for extraction)
        full_content = self._combine_messages(messages_to_process)

        # COMPREHENSIVE EXTRACTION (on new messages only)
        # Extract file paths early (needed for project tracking)
        file_paths = self._extract_file_paths(full_content)

        # Use robust extractor if available, fallback to legacy
        if ROBUST_EXTRACTION_ENABLED:
            robust_ex = RobustExtractor()
            extracted = {
                "facts": self._extract_facts(full_content, messages_to_process),
                "entities": self._extract_entities(full_content),
                "file_paths": file_paths,
                "actions_taken": robust_ex.extract_actions(full_content, messages_to_process),
                "decisions_made": robust_ex.extract_decisions(full_content, messages_to_process),
                "problems_solved": robust_ex.extract_problems_solutions(full_content, messages_to_process),
                "code_snippets": robust_ex.extract_code_snippets_simple(full_content, messages_to_process),
                "user_preferences": robust_ex.extract_preferences(full_content, messages_to_process),
                "technical_details": self._extract_technical_details(full_content),
                "goals_and_intentions": robust_ex.extract_goals(full_content, messages_to_process),
                "learnings": self._extract_learnings(full_content, messages_to_process)
            }
        else:
            # Legacy extraction
            extracted = {
                "facts": self._extract_facts(full_content, messages_to_process),
                "entities": self._extract_entities(full_content),
                "file_paths": file_paths,
                "actions_taken": self._extract_actions(full_content, messages_to_process),
                "decisions_made": self._extract_decisions(full_content, messages_to_process),
                "problems_solved": self._extract_problems_solutions(full_content, messages_to_process),
                "code_snippets": self._extract_code_snippets(full_content, messages_to_process),
                "user_preferences": self._extract_preferences(full_content, messages_to_process),
                "technical_details": self._extract_technical_details(full_content),
                "goals_and_intentions": self._extract_goals(full_content, messages_to_process),
                "learnings": self._extract_learnings(full_content, messages_to_process)
            }

        # NEW: Process corrections (Agent 2 integration)
        corrections_found = []
        if CORRECTIONS_ENABLED:
            try:
                corrections_found = process_conversation_for_corrections(
                    messages_to_process,
                    conv_id
                )
            except Exception as e:
                print(f"Warning: Correction processing failed: {e}")

        # NEW: Process projects (Agent 3 integration)
        project_info = {}
        if PROJECT_TRACKING_ENABLED:
            try:
                project_info = process_conversation_for_projects(
                    messages_to_process,
                    conv_id,
                    file_paths=[fp["path"] for fp in file_paths]
                )
            except Exception as e:
                print(f"Warning: Project tracking failed: {e}")

        # Phase 3: Add confidence levels with source-based scoring and provenance
        # Phase 6: Add provenance tracking for facts
        contradictions_detected = []

        for fact in extracted["facts"]:
            fact_id = fact.get("fact_id", "unknown")
            fact_content = fact.get("content", "")
            fact_context = fact.get("context_snippet", "")
            fact_role = fact.get("source_role", "user")
            message_idx = fact.get("source_message_index")

            if self.confidence_tracker and CONFIDENCE_TRACKING_ENABLED:
                # Detect source based on role and context
                source = self.confidence_tracker.detect_source(
                    content=fact_content,
                    context=fact_context,
                    role=fact_role
                )

                # Assign confidence with tracking and provenance
                confidence = self.confidence_tracker.assign_confidence(
                    fact_id=fact_id,
                    source=source,
                    content=fact_content,
                    context=fact_context,
                    reason=f"Extracted from {fact.get('type', 'unknown')} pattern"
                )

                fact["confidence"] = confidence
                fact["confidence_source"] = source.name_str
            else:
                # Fallback to legacy string-based confidence
                fact["confidence"] = self._determine_fact_confidence(fact, full_content)

            # Phase 6: Create provenance source chain for each fact
            if self.provenance_tracker and PROVENANCE_TRACKING_ENABLED:
                try:
                    # Detect how the fact was learned
                    learned_type = detect_learned_type(
                        content=fact_content,
                        role=fact_role,
                        context=fact_context
                    )

                    # Create source chain
                    self.provenance_tracker.create_source_chain(
                        fact_id=fact_id,
                        conversation_id=conv_id,
                        learned_type=learned_type,
                        message_index=message_idx,
                        context_snippet=fact_context,
                        extracted_by="pattern"
                    )

                    # Check for contradictions
                    contradictions = self.provenance_tracker.detect_contradictions(
                        new_fact_id=fact_id,
                        new_fact_content=fact_content
                    )
                    if contradictions:
                        contradictions_detected.extend(contradictions)
                        fact["has_contradictions"] = True
                        fact["contradiction_count"] = len(contradictions)
                except Exception as e:
                    print(f"Warning: Provenance tracking failed for {fact_id}: {e}")

        # Extract user personal profile data if enabled
        user_profile_data = None
        if self.user_extractor:
            try:
                user_profile_data = self.user_extractor.extract_all(messages_to_process, conv_id)
                extracted["user_profile"] = user_profile_data
            except Exception as e:
                print(f"Warning: User profile extraction failed: {e}")
                extracted["user_profile"] = None

        # NEW: Extract emotional patterns and communication style (Agent 16)
        emotional_patterns = None
        communication_style = None
        try:
            from emotional_detector import EmotionalDetector
            from user_profile_manager import UserProfileManager

            detector = EmotionalDetector()

            # Build temp conversation structure for detector
            temp_conv = {"messages": messages_to_process}

            # Detect emotional patterns
            emotional_patterns = detector.detect_emotional_patterns(temp_conv)
            communication_style = detector.detect_communication_style(temp_conv)

            # Update user profile with emotional context
            try:
                profile_manager = UserProfileManager()
                profile = profile_manager.load_profile()

                # Update emotional patterns
                if emotional_patterns:
                    profile = profile_manager.update_emotional_patterns(profile, emotional_patterns)

                # Update communication style
                if communication_style:
                    profile = profile_manager.update_communication_style(profile, communication_style)

                profile_manager.save_profile(profile)

                # Store in extracted data
                extracted["emotional_patterns"] = emotional_patterns
                extracted["communication_style"] = communication_style
            except Exception as e:
                print(f"Warning: User profile emotional update failed: {e}")
        except Exception as e:
            print(f"Warning: Emotional detection failed: {e}")

        # Phase 5: Apply privacy filter - detect and redact secrets
        privacy_metadata = {"enabled": False, "secrets_redacted": 0, "sensitivity_level": "public"}
        if self.privacy_filter and PRIVACY_FILTER_ENABLED:
            try:
                # Redact secrets from full content
                redaction_result = self.privacy_filter.redact(full_content)
                full_content = redaction_result.redacted_text
                privacy_metadata = {
                    "enabled": True,
                    "secrets_redacted": redaction_result.redaction_count,
                    "sensitivity_level": redaction_result.sensitivity_level.value,
                    "secret_types": [s.secret_type.value for s in redaction_result.secrets_found]
                }

                # Process each fact through privacy filter
                for fact in extracted.get("facts", []):
                    fact_content = fact.get("content", "")
                    if fact_content:
                        processed_fact = self.privacy_filter.process_fact(fact)
                        fact.update(processed_fact)

                # Redact messages content
                for msg in messages_to_process:
                    msg_content = msg.get("content", "")
                    if msg_content:
                        msg_result = self.privacy_filter.redact(msg_content)
                        msg["content"] = msg_result.redacted_text

                # Process code snippets
                for snippet in extracted.get("code_snippets", []):
                    snippet_content = snippet.get("code", "")
                    if snippet_content:
                        snippet_result = self.privacy_filter.redact(snippet_content)
                        snippet["code"] = snippet_result.redacted_text
                        if snippet_result.redaction_count > 0:
                            snippet["contains_redacted"] = True

                if privacy_metadata["secrets_redacted"] > 0:
                    print(f"[Privacy] Redacted {privacy_metadata['secrets_redacted']} secrets from conversation {conv_id}")
            except Exception as e:
                print(f"Warning: Privacy filtering failed: {e}")

        # Phase 1 Enhancement: Filter out low-confidence facts (threshold: 0.60)
        MIN_FACT_CONFIDENCE = 0.60
        original_fact_count = len(extracted.get("facts", []))
        filtered_facts = []
        for fact in extracted.get("facts", []):
            confidence = fact.get("confidence", 0.5)
            # Handle both numeric and string confidence values
            if isinstance(confidence, str):
                # Legacy string values - convert to numeric
                confidence_map = {"high": 0.9, "medium": 0.6, "low": 0.4}
                confidence = confidence_map.get(confidence.lower(), 0.5)
            if confidence >= MIN_FACT_CONFIDENCE:
                filtered_facts.append(fact)
        extracted["facts"] = filtered_facts
        if original_fact_count != len(filtered_facts):
            print(f"[Quality] Filtered {original_fact_count - len(filtered_facts)} low-confidence facts ({original_fact_count} -> {len(filtered_facts)})")

        # Build complete conversation record (need timestamp for temporal classification)
        conversation = {
            "id": conv_id,
            "timestamp": datetime.now().isoformat(),
            "type": "conversation",

            # Raw data (only the messages being saved in this segment)
            "messages": messages_to_process,
            "content": full_content,

            # Extracted structured data
            "extracted_data": extracted,

            # Placeholder for metadata (will be enhanced below)
            "metadata": {},

            # Placeholder for search index
            "search_index": {},

            # Phase 5: Privacy metadata
            "privacy": privacy_metadata
        }

        # NEW: Add temporal classification (Agent 8)
        if self.temporal_classifier:
            conversation['metadata']['temporal_category'] = self.temporal_classifier.classify_conversation(conversation)
        else:
            conversation['metadata']['temporal_category'] = 'RECENT'

        # NEW: Extract preferences and emotional context (Agent 8)
        if self.preference_manager:
            try:
                self.preference_manager.extract_preferences_from_conversation(conversation)
            except Exception as e:
                print(f"Warning: Preference extraction failed: {e}")

        # Enhanced metadata (with new corrections, projects, threading info)
        enhanced_metadata = self._create_enhanced_metadata(
            full_content, messages_to_process, extracted, metadata or {},
            corrections_found=corrections_found,
            project_info=project_info,
            session_id=session_id,
            conv_id=conv_id
        )

        # Merge temporal category into enhanced metadata
        if 'temporal_category' in conversation['metadata']:
            enhanced_metadata['temporal_category'] = conversation['metadata']['temporal_category']

        # Phase 6: Add contradiction information to metadata
        if contradictions_detected:
            enhanced_metadata['contradictions_detected'] = len(contradictions_detected)
            enhanced_metadata['contradiction_pairs'] = [
                {"fact_a": c.fact_id_a, "fact_b": c.fact_id_b, "type": c.contradiction_type}
                for c in contradictions_detected[:5]  # Limit to first 5
            ]

        # Search index
        search_index = self._create_search_index(full_content, extracted)

        # Safety-net device tagging: ensure every conversation gets a device_tag
        if "device_tag" not in enhanced_metadata:
            try:
                from device_registry import get_device_metadata_for_conversation
                device_meta = get_device_metadata_for_conversation()
                enhanced_metadata["device_tag"] = device_meta.get("device_tag", "unknown")
                enhanced_metadata["device"] = device_meta
            except Exception:
                import platform
                import socket
                enhanced_metadata["device_tag"] = "windows_pc" if platform.system() == "Windows" else "linux_pc"
                enhanced_metadata["device_hostname"] = socket.gethostname()

        # Update conversation record with final metadata and search index
        conversation["metadata"] = enhanced_metadata
        conversation["search_index"] = search_index

        # Ensure directories exist before saving
        if not self._ensure_directories():
            raise RuntimeError("Could not create required directories on NAS")

        # Upsert: if conversation file exists and has more/equal messages, update metadata only
        conv_file = self.conversations_path / f"{conv_id}.json"
        if conv_file.exists():
            try:
                with open(conv_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                existing_msg_count = len(existing.get("messages", []))
                if len(messages) <= existing_msg_count:
                    # Existing version is equal or more complete — just update metadata
                    existing["metadata"].update(enhanced_metadata or {})
                    with open(conv_file, "w", encoding="utf-8") as f:
                        json.dump(existing, f, indent=2, ensure_ascii=False)
                    return conv_id
            except (json.JSONDecodeError, KeyError):
                pass  # Corrupted file, overwrite it

        # Save conversation (new or has more messages)
        with open(conv_file, "w", encoding="utf-8") as f:
            json.dump(conversation, f, indent=2, ensure_ascii=False)

        # AGENT 9: Extract and index code snippets
        try:
            from code_indexer import CodeIndexer
            code_indexer = CodeIndexer(base_path=str(self.base_path))
            code_snippets = code_indexer.extract_code_from_conversation(conversation)
            if code_snippets:
                code_indexer.save_snippets(code_snippets, conv_id)
        except Exception as e:
            print(f"Warning: Code indexing failed: {e}")

        # Update knowledge base indexes
        self._update_knowledge_base(conv_id, extracted, enhanced_metadata)
        self._update_entity_database(conv_id, extracted["entities"])
        self._update_timeline(conv_id, extracted, enhanced_metadata)

        # Update user profile if personal data was extracted
        if user_profile_data and user_profile_data.get('is_personal_conversation'):
            try:
                update_user_profile(conv_id, user_profile_data, str(self.base_path))
                print(f"[User] Updated profile with data from conversation {conv_id}")
            except Exception as e:
                print(f"Warning: Failed to update user profile: {e}")

        # Update session state for incremental saves
        if session_id:
            self._update_session_state(
                session_id=session_id,
                message_count=len(messages),  # Total messages (not just processed)
                last_conversation_id=conv_id,
                segment_count=enhanced_metadata.get("segment_number", 1) if incremental else 1
            )

        return conv_id

    def _combine_messages(self, messages: List[Dict[str, str]]) -> str:
        """Combine all messages into searchable text"""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    def _get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get the current state for an incremental session"""
        try:
            state_file = self.session_states_path / f"{session_id}.json"
            if state_file.exists():
                with open(state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"WARNING: Could not read session state: {e}")

        return {
            "session_id": session_id,
            "message_count": 0,
            "segment_count": 0,
            "conversation_ids": [],
            "created_at": datetime.now().isoformat()
        }

    def _update_session_state(self, session_id: str, message_count: int,
                              last_conversation_id: str, segment_count: int):
        """Update session state after saving a conversation segment"""
        state_file = self.session_states_path / f"{session_id}.json"

        # Load existing or create new
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        else:
            state = {
                "session_id": session_id,
                "conversation_ids": [],
                "created_at": datetime.now().isoformat()
            }

        # Update state
        state["message_count"] = message_count
        state["last_conversation_id"] = last_conversation_id
        state["segment_count"] = segment_count
        state["updated_at"] = datetime.now().isoformat()

        # Track all conversation IDs for this session
        if last_conversation_id not in state["conversation_ids"]:
            state["conversation_ids"].append(last_conversation_id)

        # Save
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get all conversation segments for a session (for reconstruction)"""
        state = self._get_session_state(session_id)
        conversations = []
        for conv_id in state.get("conversation_ids", []):
            conv_file = self.conversations_path / f"{conv_id}.json"
            if conv_file.exists():
                with open(conv_file, "r", encoding="utf-8") as f:
                    conversations.append(json.load(f))
        return conversations

    def get_full_conversation(self, session_id: str) -> List[Dict[str, str]]:
        """Reconstruct full conversation from all segments"""
        segments = self.get_session_history(session_id)
        all_messages = []
        for segment in segments:
            all_messages.extend(segment.get("messages", []))
        return all_messages

    def _is_valid_fact(self, content: str) -> bool:
        """
        Validate fact content quality. Reject garbage facts.
        Returns True if fact is valid, False if should be rejected.

        Enhanced (Phase 7) with:
        - Stricter noise filtering
        - Technical content requirements
        - Better filler/routine detection
        """
        if not content:
            return False

        # Reject too short facts (less than 30 chars or 6 words)
        words = content.split()
        if len(content) < 30 or len(words) < 6:
            return False

        content_lower = content.lower()

        # Reject conversation fragments (these are process descriptions, not facts)
        conversation_patterns = [
            r"^(?:before\s+i|let\s+me|i'?ll|i\s+will|i'm\s+going\s+to|first\s+i|now\s+i)",
            r"^(?:looking\s+at|checking|reading|searching|analyzing)",
            r"^(?:here'?s?\s+what|this\s+is\s+what|the\s+output\s+shows)",
            r"^(?:based\s+on|according\s+to|as\s+you\s+can\s+see)",
        ]
        for pattern in conversation_patterns:
            if re.match(pattern, content_lower):
                return False

        # Reject tool call content
        tool_patterns = [
            r"\[Tool:",
            r"```(?:json|python|bash|javascript|typescript)",
            r"<",
            r"function_call",
            r"<tool_use>",
        ]
        for pattern in tool_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False

        # Reject non-declarative statements (questions, commands)
        if content.strip().endswith("?"):
            return False
        if content_lower.startswith(("please ", "can you ", "could you ", "would you ")):
            return False

        # Reject common filler phrases that aren't facts
        filler_patterns = [
            r"^(?:okay|ok|sure|yes|no|right|got\s+it|understood|alright)",
            r"^(?:i\s+see|i\s+understand|makes\s+sense|noted)",
            r"^(?:great|perfect|excellent|awesome|nice|good\s+job)",
        ]
        for pattern in filler_patterns:
            if re.match(pattern, content_lower):
                return False

        # NEW (Phase 7): Reject routine/trivial conversation patterns
        routine_patterns = [
            r"^(?:hi|hello|hey|greetings)\b",
            r"^(?:thanks?|thank\s+you)\b",
            r"^(?:bye|goodbye|see\s+you)\b",
            r"^(?:sounds?\s+good|that\s+works?)\b",
            r"^(?:let'?s?\s+(?:go|do|start|try))\b",
            r"^(?:one\s+moment|hold\s+on|wait)\b",
            r"^(?:hmm|uhh?|ahh?|well)\b",
        ]
        for pattern in routine_patterns:
            if re.match(pattern, content_lower):
                return False

        # NEW (Phase 7): Reject assistant process narration
        assistant_narration = [
            r"^(?:i'?ll\s+(?:now|first|start|begin|check))",
            r"^(?:let\s+me\s+(?:check|look|see|examine|read))",
            r"^(?:checking|looking|reading|examining|searching)",
            r"^(?:i\s+(?:see|found|noticed)\s+that)",
            r"^(?:it\s+(?:looks|seems|appears)\s+like)",
        ]
        for pattern in assistant_narration:
            if re.match(pattern, content_lower):
                return False

        # NEW (Phase 7): Require SOME technical substance
        # Facts should contain at least one technical indicator
        technical_indicators = [
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # IP
            r':\d{2,5}\b',  # Port
            r'[A-Z]:\\',  # Windows path
            r'/(?:home|usr|var|etc|opt)/',  # Unix path
            r'\.(?:py|js|ts|json|yaml|yml|toml|ini|cfg|conf|sh|bat|ps1)\b',  # File extensions
            r'\b(?:python|node|docker|git|npm|pip|redis|postgres)\b',  # Tech keywords
            r'\b(?:config|server|api|database|memory|storage)\b',  # Infra keywords
            r'\bv?\d+\.\d+(?:\.\d+)?\b',  # Version numbers
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase (class names)
            r'\b[a-z]+_[a-z]+\b',  # snake_case (variable names)
        ]

        has_technical = any(re.search(pattern, content, re.IGNORECASE)
                          for pattern in technical_indicators)

        # If no technical content and fact is short-ish, reject
        if not has_technical and len(content) < 80:
            return False

        return True

    def _is_valid_problem_solution(self, problem: str, solution: str) -> bool:
        """Reject garbage problem/solution pairs."""
        if not problem or len(problem) < 20 or len(problem.split()) < 4:
            return False
        if solution and solution != "Not explicitly stated" and solution != "Not yet resolved":
            if len(solution) < 20 or len(solution.split()) < 4:
                return False
        # Reject if problem is just a sentence fragment (no technical substance)
        problem_lower = problem.lower()
        has_substance = any(w in problem_lower for w in [
            'error', 'fail', 'crash', 'broken', 'issue', 'bug', 'not working',
            'cannot', 'unable', 'timeout', 'missing', 'wrong', 'incorrect',
            'invalid', 'exception', 'traceback', 'refused', 'denied'
        ])
        if not has_substance:
            has_technical = any(c in problem for c in ['/', '\\', '.py', '.js', '()', '::', 'http'])
            if not has_technical:
                return False
        return True

    def _should_skip_message_extraction(self, message: Dict) -> bool:
        """
        Check if a message should be skipped for fact extraction.
        (Phase 7 Enhancement)

        Returns True if message is too short, routine, or lacks substance.
        """
        content = message.get("content", "")
        role = message.get("role", "")

        # Skip very short messages (< 50 chars)
        if len(content) < 50:
            return True

        content_lower = content.lower()

        # Skip pure greetings/acknowledgments
        routine_only = [
            r"^(?:hi|hello|hey|thanks?|ok(?:ay)?|sure|yes|no|alright)[.!]?$",
            r"^(?:got\s+it|sounds?\s+good|makes\s+sense|understood)[.!]?$",
            r"^(?:perfect|great|awesome|nice|cool)[.!]?$",
        ]
        for pattern in routine_only:
            if re.match(pattern, content_lower.strip()):
                return True

        # Skip tool-heavy assistant messages (mostly tool calls, not content)
        if role == "assistant":
            tool_indicators = content.count("[Tool:") + content.count("<tool_use>")
            text_ratio = len(re.sub(r'\[Tool:.*?\]|<tool_use>.*?</tool_use>', '', content, flags=re.DOTALL)) / max(len(content), 1)
            if tool_indicators > 2 and text_ratio < 0.3:
                return True

        return False

    def _extract_facts(self, content: str, messages: List[Dict]) -> List[Dict]:
        """Extract factual statements, learnings, and discoveries"""
        facts = []

        # Technical facts patterns
        technical_patterns = [
            (r"(.+?)\s+(?:doesn't|does not|won't|will not)\s+(?:work|trigger|run|execute|support)(.+)", "technical_limitation"),
            (r"(.+?)\s+(?:works|triggers|runs|executes|supports)(.+)", "technical_capability"),
            (r"(?:discovered|found|learned)\s+(?:that\s+)?(.+)", "discovery"),
            (r"(?:the|a)\s+(.+?)\s+is\s+(?:located|stored|saved)\s+(?:at|in)\s+(.+)", "location"),
            (r"(?:uses?|using|utilized?)\s+(.+?)\s+(?:for|to)\s+(.+)", "usage"),
            (r"(?:configured|set up|installed)\s+(.+)", "configuration"),
        ]

        for i, msg in enumerate(messages):
            # Phase 7: Skip messages that lack substance
            if self._should_skip_message_extraction(msg):
                continue

            content_text = msg.get("content", "")
            role = msg.get("role", "user")

            for pattern, fact_type in technical_patterns:
                matches = re.finditer(pattern, content_text, re.IGNORECASE)
                for match in matches:
                    fact_content = match.group(0).strip()

                    # Generate unique fact ID
                    fact_id = hashlib.sha256(
                        f"{fact_type}:{fact_content}".encode()
                    ).hexdigest()[:16]

                    # Get context (surrounding messages)
                    context = ""
                    if i > 0:
                        context = messages[i-1].get("content", "")

                    # Validate fact content quality
                    if not self._is_valid_fact(fact_content):
                        continue

                    facts.append({
                        "fact_id": fact_id,
                        "type": fact_type,
                        "content": fact_content,
                        "confidence": "high",  # Will be replaced by numeric score below
                        "source_role": role,
                        "source_message_index": i,
                        "context_snippet": context[:200] if context else "",
                        "extracted_at": datetime.now().isoformat()
                    })

        # Remove duplicates
        seen = set()
        unique_facts = []
        for fact in facts:
            fact_key = f"{fact['type']}:{fact['content']}"
            if fact_key not in seen:
                seen.add(fact_key)
                unique_facts.append(fact)

        return unique_facts

    def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract entities: people, tools, technologies, servers, networks"""
        entities = {
            "people": set(),
            "tools": set(),
            "technologies": set(),
            "servers": set(),
            "networks": set(),
            "companies": set(),
            "os_platforms": set()
        }

        # People (usernames, names)
        people_pattern = r"(?:user(?:name)?):\s*(\w+)"
        entities["people"].update(re.findall(people_pattern, content, re.IGNORECASE))

        # Common tools
        tool_keywords = [
            "Claude", "Python", "PowerShell", "Bash", "Git", "Docker",
            "MCP", "Obsidian", "Synology", "NAS", "PostgreSQL", "Redis",
            "npm", "pip", "VS Code", "Terminal", "CMD", "JSON", "JSONL"
        ]
        for tool in tool_keywords:
            if re.search(r'\b' + re.escape(tool) + r'\b', content, re.IGNORECASE):
                entities["tools"].add(tool)

        # Technologies
        tech_keywords = [
            "JSON", "JSONL", "JSON-RPC", "stdio", "SMB", "TCP/IP",
            "HTTP", "HTTPS", "API", "REST", "WebSocket", "SSH"
        ]
        for tech in tech_keywords:
            if re.search(r'\b' + re.escape(tech) + r'\b', content, re.IGNORECASE):
                entities["technologies"].add(tech)

        # Server names (MCP servers, etc.)
        server_pattern = r'(?:server|mcp)[\s-]*(?:name)?[\s:]+([a-z0-9_-]+)'
        entities["servers"].update(re.findall(server_pattern, content, re.IGNORECASE))

        # Network addresses
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        # Only match hostnames that look like actual server names (not random words)
        hostname_pattern = r'\b(?:NAS|SERVER|HOST|VM|DOCKER|SYNOLOGY|PROXMOX)[A-Z0-9-]*\b'
        entities["networks"].update(re.findall(ip_pattern, content))
        entities["networks"].update(re.findall(hostname_pattern, content, re.IGNORECASE))

        # OS/Platforms
        os_keywords = ["Windows", "Linux", "macOS", "Unix"]
        for os_name in os_keywords:
            if re.search(r'\b' + re.escape(os_name) + r'\b', content, re.IGNORECASE):
                entities["os_platforms"].add(os_name)

        # === FILTER OUT FALSE POSITIVES ===

        # Filter invalid people (pronouns, single letters, common words)
        INVALID_PEOPLE = {
            'i', 'me', 'my', 'you', 'your', 'we', 'our', 'they', 'them',
            'user', 'assistant', 'a', 'an', 'the', 'it', 'is', 'was',
            'have', 'has', 'had', 'do', 'does', 'did', 'be', 'been',
            'ok', 'yes', 'no', 'hi', 'hello', 'hey', 'thanks', 'thank'
        }
        entities["people"] = {
            p for p in entities["people"]
            if p.lower() not in INVALID_PEOPLE and len(p) > 1
        }

        # Filter invalid networks (too short, just numbers, common words)
        entities["networks"] = {
            n for n in entities["networks"]
            if len(n) > 2
            and not n.isdigit()
            and n.lower() not in {'the', 'and', 'for', 'but', 'not'}
        }

        # Filter invalid servers
        entities["servers"] = {
            s for s in entities["servers"]
            if len(s) > 2 and s.lower() not in {'the', 'a', 'an', 'is'}
        }

        # Convert sets to sorted lists
        return {k: sorted(list(v)) for k, v in entities.items()}

    def _extract_file_paths(self, content: str) -> List[Dict]:
        """Extract all file paths with context"""
        paths = []

        # Windows paths
        win_pattern = r'([A-Z]:\\(?:[^\s\\/:*?"<>|]+\\)*[^\s\\/:*?"<>|]*)'
        # Unix paths
        unix_pattern = r'(/(?:[^\s/:*?"<>|]+/)*[^\s/:*?"<>|]*)'
        # Relative paths
        rel_pattern = r'\.{1,2}/(?:[^\s/:*?"<>|]+/)*[^\s/:*?"<>|]*'

        for pattern, path_type in [
            (win_pattern, "windows"),
            (unix_pattern, "unix"),
            (rel_pattern, "relative")
        ]:
            matches = re.finditer(pattern, content)
            for match in matches:
                path = match.group(1) if path_type != "relative" else match.group(0)

                # Get context around the path
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end].strip()

                # Determine purpose from context
                purpose = "unknown"
                if re.search(r'\b(?:config|settings|configuration)\b', context, re.IGNORECASE):
                    purpose = "configuration"
                elif re.search(r'\b(?:script|code|program)\b', context, re.IGNORECASE):
                    purpose = "script"
                elif re.search(r'\b(?:log|logging)\b', context, re.IGNORECASE):
                    purpose = "log"
                elif re.search(r'\b(?:data|storage|save)\b', context, re.IGNORECASE):
                    purpose = "data_storage"
                elif re.search(r'\b(?:hook|command)\b', context, re.IGNORECASE):
                    purpose = "hook_or_command"

                paths.append({
                    "path": path,
                    "type": path_type,
                    "purpose": purpose,
                    "context": context,
                    "extracted_at": datetime.now().isoformat()
                })

        # Remove duplicates
        seen = set()
        unique_paths = []
        for path_info in paths:
            if path_info["path"] not in seen:
                seen.add(path_info["path"])
                unique_paths.append(path_info)

        return unique_paths

    def _extract_actions(self, content: str, messages: List[Dict]) -> List[Dict]:
        """Extract actions taken: files created, configs modified, commands run"""
        actions = []

        action_patterns = [
            (r'(?:created|creating|create|made|wrote)\s+(?:file|script|command)?(?:\s+at)?\s+([^\s]+)', "create_file"),
            (r'(?:modified|updated|changed|edited)\s+([^\s]+)', "modify_file"),
            (r'(?:deleted|removed)\s+([^\s]+)', "delete_file"),
            (r'(?:installed|added)\s+([^\s]+)', "install"),
            (r'(?:configured|set up|setup)\s+([^\s]+)', "configure"),
            (r'(?:ran|executed|running)\s+([^\s]+)', "execute"),
            (r'(?:fixed|resolved|solved)\s+(.+)', "fix"),
        ]

        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant":  # Focus on AI actions
                content_text = msg.get("content", "")

                for pattern, action_type in action_patterns:
                    matches = re.finditer(pattern, content_text, re.IGNORECASE)
                    for match in matches:
                        actions.append({
                            "action_type": action_type,
                            "target": match.group(1).strip() if match.lastindex >= 1 else match.group(0),
                            "description": match.group(0).strip(),
                            "message_index": i,
                            "timestamp": datetime.now().isoformat()
                        })

        return actions

    def _extract_decisions(self, content: str, messages: List[Dict]) -> List[Dict]:
        """Extract decisions made and reasoning"""
        decisions = []

        decision_patterns = [
            r'(?:decided to|choosing to|opted to|going to use)\s+(.+?)(?:\.|because|since|,)',
            r'(?:instead of|rather than)\s+(.+?)(?:,|\.|\s+we)',
            r'(?:the (?:best|better|recommended))\s+(?:approach|solution|method|way)\s+is\s+(.+?)(?:\.|,)',
        ]

        for pattern in decision_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Get surrounding context
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end].strip()

                decisions.append({
                    "decision": match.group(1).strip(),
                    "context": context,
                    "extracted_at": datetime.now().isoformat()
                })

        return decisions

    def _extract_problems_solutions(self, content: str, messages: List[Dict]) -> List[Dict]:
        """Extract problems encountered and their solutions"""
        problems = []

        # Look for problem descriptions (require minimum 20 chars of content)
        problem_patterns = [
            r'(?:problem|issue|error|bug):\s*(.{20,}?)(?:\n\n|\n(?=[A-Z])|\Z)',
            r'(?:not working|doesn\'t work|won\'t work|failed):\s*(.{20,}?)(?:\n\n|\n(?=[A-Z])|\Z)',
            r'(?:the\s+)?(.{20,}?)\s+(?:is not working|doesn\'t work)',
        ]

        solution_patterns = [
            r'(?:solution|fix|resolved):\s*(.+?)(?:\n|$)',
            r'(?:fixed by|solved by|resolved by)\s+(.+?)(?:\n|$)',
            r'(?:to fix this|to solve this),?\s+(.+?)(?:\n|$)',
        ]

        # Try to pair problems with solutions
        for i, msg in enumerate(messages):
            content_text = msg.get("content", "")

            # Look for problem-solution pairs
            if "problem" in content_text.lower() or "issue" in content_text.lower():
                problem_match = None
                for pattern in problem_patterns:
                    match = re.search(pattern, content_text, re.IGNORECASE)
                    if match:
                        problem_match = match.group(1).strip()
                        break

                if problem_match:
                    # Look for solution in next few messages
                    solution_match = None
                    for j in range(i, min(i + 3, len(messages))):
                        sol_content = messages[j].get("content", "")
                        for pattern in solution_patterns:
                            match = re.search(pattern, sol_content, re.IGNORECASE)
                            if match:
                                solution_match = match.group(1).strip()
                                break
                        if solution_match:
                            break

                    if not self._is_valid_problem_solution(problem_match, solution_match):
                        continue
                    problems.append({
                        "problem": problem_match,
                        "solution": solution_match or "Not yet resolved",
                        "status": "resolved" if solution_match else "open",
                        "discovered_at": datetime.now().isoformat()
                    })

        return problems

    def _extract_code_snippets(self, content: str, messages: List[Dict]) -> List[Dict]:
        """Extract code snippets with language and purpose"""
        snippets = []

        # Code block pattern (markdown style)
        code_block_pattern = r'```(\w+)?\n(.*?)```'
        matches = re.finditer(code_block_pattern, content, re.DOTALL)

        for match in matches:
            language = match.group(1) or "unknown"
            code = match.group(2).strip()

            # Try to determine purpose from surrounding context
            start = max(0, match.start() - 100)
            context_before = content[start:match.start()]

            purpose = "general"
            if "config" in context_before.lower():
                purpose = "configuration"
            elif "hook" in context_before.lower():
                purpose = "hook"
            elif "test" in context_before.lower():
                purpose = "testing"
            elif "fix" in context_before.lower():
                purpose = "fix"

            snippets.append({
                "language": language,
                "code": code,
                "purpose": purpose,
                "length": len(code),
                "extracted_at": datetime.now().isoformat()
            })

        # Also look for inline file paths in code
        if snippets:
            for snippet in snippets:
                if snippet["language"] in ["json", "powershell", "python"]:
                    snippet["referenced_files"] = re.findall(
                        r'[A-Z]:\\(?:[^\s\\/:*?"<>|]+\\)*[^\s\\/:*?"<>|]*',
                        snippet["code"]
                    )

        return snippets

    def _extract_preferences(self, content: str, messages: List[Dict]) -> List[Dict]:
        """Extract user preferences, likes, dislikes"""
        preferences = []

        preference_patterns = [
            (r'i want\s+(?:to\s+)?(.+?)(?:\.|,|\n)', "want"),
            (r'i (?:prefer|like)\s+(.+?)(?:\.|,|\n)', "prefer"),
            (r'i (?:don\'t want|hate|dislike)\s+(.+?)(?:\.|,|\n)', "dislike"),
            (r'(?:my goal is|i\'m trying to)\s+(.+?)(?:\.|,|\n)', "goal"),
        ]

        for msg in messages:
            if msg.get("role") == "user":
                content_text = msg.get("content", "")

                for pattern, pref_type in preference_patterns:
                    matches = re.finditer(pattern, content_text, re.IGNORECASE)
                    for match in matches:
                        preferences.append({
                            "type": pref_type,
                            "preference": match.group(1).strip(),
                            "extracted_at": datetime.now().isoformat()
                        })

        return preferences

    def _extract_technical_details(self, content: str) -> Dict[str, Any]:
        """Extract technical environment details"""
        details = {
            "programming_languages": set(),
            "versions": {},
            "ports": [],
            "ips": [],
            "credentials": {},
            "capacities": {}
        }

        # Programming languages mentioned
        languages = ["Python", "JavaScript", "PowerShell", "Bash", "JSON", "SQL"]
        for lang in languages:
            if re.search(r'\b' + re.escape(lang) + r'\b', content, re.IGNORECASE):
                details["programming_languages"].add(lang)

        # Version numbers
        version_pattern = r'(\w+)\s+(?:version\s+)?(\d+\.[\d.]+)'
        for match in re.finditer(version_pattern, content, re.IGNORECASE):
            details["versions"][match.group(1)] = match.group(2)

        # Port numbers
        port_pattern = r':(\d{2,5})\b'
        details["ports"] = list(set(re.findall(port_pattern, content)))

        # IP addresses
        ip_pattern = r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'
        details["ips"] = list(set(re.findall(ip_pattern, content)))

        # Storage capacities
        capacity_pattern = r'(\d+)\s*(TB|GB|MB|KB)'
        for match in re.finditer(capacity_pattern, content, re.IGNORECASE):
            details["capacities"][f"{match.group(1)}{match.group(2)}"] = {
                "value": int(match.group(1)),
                "unit": match.group(2).upper()
            }

        # Convert sets to lists
        details["programming_languages"] = sorted(list(details["programming_languages"]))

        return details

    def _extract_goals(self, content: str, messages: List[Dict]) -> List[Dict]:
        """Extract user's goals and intentions"""
        goals = []

        goal_patterns = [
            r'(?:my goal|i want to|trying to|plan to)\s+(.+?)(?:\.|,|\n)',
            r'(?:essentially|basically),?\s+(.+?)(?:\.|,|\n)',
            r'(?:the purpose is to|this is for)\s+(.+?)(?:\.|,|\n)',
        ]

        for msg in messages:
            if msg.get("role") == "user":
                content_text = msg.get("content", "")

                for pattern in goal_patterns:
                    matches = re.finditer(pattern, content_text, re.IGNORECASE)
                    for match in matches:
                        goals.append({
                            "goal": match.group(1).strip(),
                            "stated_at": datetime.now().isoformat(),
                            "priority": "high"  # Could be enhanced with sentiment analysis
                        })

        return goals

    def _extract_learnings(self, content: str, messages: List[Dict]) -> List[Dict]:
        """Extract key learnings and insights"""
        learnings = []

        learning_patterns = [
            r'(?:learned that|discovered that|found that)\s+(.+?)(?:\.|,|\n)',
            r'(?:key (?:takeaway|learning|insight)):\s*(.+?)(?:\.|,|\n)',
            r'(?:important to note|remember that)\s+(.+?)(?:\.|,|\n)',
        ]

        for pattern in learning_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                learnings.append({
                    "learning": match.group(1).strip(),
                    "confidence": "high",
                    "learned_at": datetime.now().isoformat()
                })

        return learnings

    def _create_enhanced_metadata(self, content: str, messages: List[Dict],
                                  extracted: Dict, base_metadata: Dict,
                                  corrections_found: List = None,
                                  project_info: Dict = None,
                                  session_id: str = None,
                                  conv_id: str = None) -> Dict:
        """Create comprehensive metadata with Agent 2 & 3 enhancements"""

        # Auto-generate comprehensive tags
        tags = set()

        # From entities
        if extracted["entities"]["tools"]:
            tags.update([t.lower() for t in extracted["entities"]["tools"]])
        if extracted["entities"]["technologies"]:
            tags.update([t.lower() for t in extracted["entities"]["technologies"]])

        # From actions
        action_tags = {action["action_type"] for action in extracted["actions_taken"]}
        tags.update(action_tags)

        # From content keywords
        important_keywords = [
            "configuration", "troubleshooting", "automation", "setup",
            "installation", "debugging", "optimization", "integration"
        ]
        for keyword in important_keywords:
            if keyword in content.lower():
                tags.add(keyword)

        # Determine topics
        topics = set()
        if any(word in content.lower() for word in ["config", "settings", "setup"]):
            topics.add("configuration")
        if any(word in content.lower() for word in ["problem", "issue", "error", "bug", "fix"]):
            topics.add("troubleshooting")
        if any(word in content.lower() for word in ["automat", "hook", "trigger"]):
            topics.add("automation")
        if any(word in content.lower() for word in ["nas", "storage", "memory", "save"]):
            topics.add("data-storage")
        if any(word in content.lower() for word in ["mcp", "server", "tool"]):
            topics.add("mcp-integration")

        # Determine complexity
        complexity = "medium"
        if len(extracted["file_paths"]) > 5 or len(extracted["actions_taken"]) > 10:
            complexity = "high"
        elif len(messages) < 5:
            complexity = "low"

        # Determine importance
        importance = "medium"
        if any(word in content.lower() for word in ["critical", "important", "essential"]):
            importance = "high"
        if len(extracted["user_preferences"]) > 3 or len(extracted["goals_and_intentions"]) > 2:
            importance = "high"

        # Build base metadata
        metadata = {
            **base_metadata,
            "tags": sorted(list(tags)),
            "topics": sorted(list(topics)),
            "complexity": complexity,
            "importance": importance,
            "message_count": len(messages),
            "word_count": len(content.split()),
            "character_count": len(content),
            "has_code": len(extracted["code_snippets"]) > 0,
            "has_problems": len(extracted["problems_solved"]) > 0,
            "has_decisions": len(extracted["decisions_made"]) > 0,
            "entity_count": sum(len(v) for v in extracted["entities"].values()),
            "file_path_count": len(extracted["file_paths"]),
            "action_count": len(extracted["actions_taken"]),
        }

        # NEW: Add corrections metadata (Agent 2)
        if corrections_found is not None:
            metadata["corrections_count"] = len(corrections_found)
            metadata["corrections"] = corrections_found

        # NEW: Add project tracking metadata (Agent 3)
        if project_info:
            metadata["projects"] = project_info.get("projects_detected", [])
            metadata["project_updates"] = project_info.get("projects_updated", [])

        # NEW: Add temporal tag
        metadata["temporal_tag"] = self._get_temporal_tag()

        # NEW: Add conversation threading
        metadata["conversation_thread"] = self._build_thread_info(session_id, conv_id)

        # Preserve autosave metadata if present
        autosave_keys = ["autosave", "hook_event", "autosave_number", "completion_status",
                        "session_complete", "saved_at"]
        for key in autosave_keys:
            if key in base_metadata:
                metadata[key] = base_metadata[key]

        return metadata

    def _create_search_index(self, content: str, extracted: Dict) -> Dict:
        """Create optimized search index"""

        # Extract all unique words (keywords)
        words = re.findall(r'\b\w{3,}\b', content.lower())
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1

        # Top keywords by frequency
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:30]
        keywords = [k for k, _ in top_keywords]

        # Generate summary (first user message + conclusion)
        summary_parts = []
        if extracted.get("problems_solved"):
            problem = extracted["problems_solved"][0]["problem"]
            solution = extracted["problems_solved"][0]["solution"]
            summary_parts.append(f"Problem: {problem}. Solution: {solution}")

        if extracted.get("goals_and_intentions"):
            goal = extracted["goals_and_intentions"][0]["goal"]
            summary_parts.append(f"Goal: {goal}")

        summary = " ".join(summary_parts) if summary_parts else content[:200]

        # Key takeaways
        takeaways = []
        if extracted.get("learnings"):
            takeaways.extend([l["learning"] for l in extracted["learnings"][:3]])
        if extracted.get("facts"):
            takeaways.extend([f["content"] for f in extracted["facts"][:3]])

        return {
            "keywords": keywords,
            "summary": summary,
            "key_takeaways": takeaways[:5],  # Top 5
            "searchable_text": content.lower(),
            "indexed_at": datetime.now().isoformat()
        }

    def _update_knowledge_base(self, conv_id: str, extracted: Dict, metadata: Dict):
        """Update central knowledge base with facts and learnings"""
        kb_file = self.facts_path / "facts.jsonl"

        with open(kb_file, "a", encoding="utf-8") as f:
            # Add facts
            for fact in extracted["facts"]:
                entry = {
                    "type": "fact",
                    "conversation_id": conv_id,
                    "timestamp": metadata.get("timestamp"),
                    **fact
                }
                f.write(json.dumps(entry) + "\n")

            # Add learnings
            for learning in extracted["learnings"]:
                entry = {
                    "type": "learning",
                    "conversation_id": conv_id,
                    "timestamp": metadata.get("timestamp"),
                    **learning
                }
                f.write(json.dumps(entry) + "\n")

    def _update_entity_database(self, conv_id: str, entities: Dict):
        """Update entity database for quick lookups"""
        for entity_type, entity_list in entities.items():
            if entity_list:
                entity_file = self.entities_path / f"{entity_type}.json"

                # Load existing
                if entity_file.exists():
                    with open(entity_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    data = {}

                # Update with new entities
                for entity in entity_list:
                    if entity not in data:
                        data[entity] = {
                            "first_seen": datetime.now().isoformat(),
                            "conversations": []
                        }
                    data[entity]["conversations"].append(conv_id)
                    data[entity]["last_seen"] = datetime.now().isoformat()

                # Save
                with open(entity_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

    def _update_timeline(self, conv_id: str, extracted: Dict, metadata: Dict):
        """Update timeline of actions and events"""
        timeline_file = self.timeline_path / f"{datetime.now():%Y-%m}.jsonl"

        with open(timeline_file, "a", encoding="utf-8") as f:
            # Add actions to timeline
            for action in extracted["actions_taken"]:
                event = {
                    "type": "action",
                    "conversation_id": conv_id,
                    "timestamp": action.get("timestamp"),
                    "action_type": action["action_type"],
                    "description": action["description"]
                }
                f.write(json.dumps(event) + "\n")

            # Add decisions to timeline
            for decision in extracted["decisions_made"]:
                event = {
                    "type": "decision",
                    "conversation_id": conv_id,
                    "timestamp": decision.get("extracted_at"),
                    "decision": decision["decision"]
                }
                f.write(json.dumps(event) + "\n")

    def _determine_fact_confidence(self, fact: dict, context: str) -> str:
        """
        Determine confidence level of a fact.

        Returns: "high", "medium", or "low"
        """
        content = fact.get("content", "").lower()

        # HIGH confidence indicators
        high_indicators = [
            "is", "are", "was", "were",  # Definitive statements
            "always", "never",  # Absolutes
            "confirmed", "verified",  # Explicit confirmation
        ]

        # LOW confidence indicators
        low_indicators = [
            "might", "maybe", "possibly", "probably",
            "i think", "seems like", "appears to",
            "could be", "not sure",
        ]

        # Check for explicit confirmation in context
        if any(phrase in context.lower() for phrase in ["yes, that's correct", "exactly", "confirmed"]):
            return "high"

        # Check low confidence
        if any(ind in content for ind in low_indicators):
            return "low"

        # Check high confidence
        if any(ind in content for ind in high_indicators):
            return "high"

        # Default medium
        return "medium"

    def _get_temporal_tag(self) -> str:
        """Get temporal tag for current conversation."""
        # This is always RECENT for new conversations
        # (Will be updated by temporal_classifier in save_conversation)
        return "RECENT"

    def _build_thread_info(self, session_id: str, conv_id: str) -> dict:
        """
        Build conversation threading information.

        Returns: {
            "session_id": str,
            "conversation_id": str,
            "parent_conversation": str or None,
            "thread_position": int
        }
        """
        thread_info = {
            "session_id": session_id,
            "conversation_id": conv_id,
            "parent_conversation": None,
            "thread_position": 0
        }

        if session_id:
            # Get session state to find parent
            session_state = self._get_session_state(session_id)
            thread_info["parent_conversation"] = session_state.get("last_conversation_id")
            thread_info["thread_position"] = session_state.get("segment_count", 0)

        return thread_info

    def _update_temporal_tags(self):
        """
        Background task to update temporal tags on old conversations.
        Call this periodically or on startup.
        """

        now = datetime.now()

        for conv_file in self.conversations_path.glob("*.json"):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conv = json.load(f)

                timestamp = datetime.fromisoformat(conv["timestamp"])
                age_days = (now - timestamp).days

                # Update temporal tag
                if age_days <= 7:
                    new_tag = "recent"
                elif age_days <= 30:
                    new_tag = "current"
                else:
                    new_tag = "historical"

                # Only update if changed
                if conv["metadata"].get("temporal_tag") != new_tag:
                    conv["metadata"]["temporal_tag"] = new_tag
                    with open(conv_file, 'w', encoding='utf-8') as f:
                        json.dump(conv, f, indent=2)
            except:
                continue

    def get_conversation_thread(self, session_id: str) -> List[Dict]:
        """
        Get all conversations in a thread.

        Returns: List of conversation dicts in chronological order
        """
        thread = []

        for conv_file in self.conversations_path.glob("*.json"):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conv = json.load(f)

                thread_info = conv.get("metadata", {}).get("conversation_thread", {})
                if thread_info.get("session_id") == session_id:
                    thread.append(conv)
            except:
                continue

        # Sort by thread position
        thread.sort(key=lambda x: x.get("metadata", {}).get("conversation_thread", {}).get("thread_position", 0))

        return thread

    def add_user_tags(self, conversation_id: str, tags: List[str]) -> bool:
        """
        Add user-defined tags to a conversation.

        Args:
            conversation_id: ID of conversation
            tags: List of tags to add

        Returns:
            Success boolean
        """
        conv_file = self.conversations_path / f"{conversation_id}.json"

        if not conv_file.exists():
            return False

        try:
            with open(conv_file, 'r', encoding='utf-8') as f:
                conv = json.load(f)

            # Add user tags (separate from auto-generated tags)
            if 'user_tags' not in conv.get('metadata', {}):
                conv.setdefault('metadata', {})['user_tags'] = []

            # Add new tags (avoid duplicates)
            for tag in tags:
                if tag not in conv['metadata']['user_tags']:
                    conv['metadata']['user_tags'].append(tag)

            # Save
            with open(conv_file, 'w', encoding='utf-8') as f:
                json.dump(conv, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"Error adding user tags: {e}")
            return False

    def add_user_note(self, conversation_id: str, note: str) -> bool:
        """
        Add a user note to a conversation.

        Args:
            conversation_id: ID of conversation
            note: User's note

        Returns:
            Success boolean
        """
        conv_file = self.conversations_path / f"{conversation_id}.json"

        if not conv_file.exists():
            return False

        try:
            with open(conv_file, 'r', encoding='utf-8') as f:
                conv = json.load(f)

            # Add user notes
            if 'user_notes' not in conv.get('metadata', {}):
                conv.setdefault('metadata', {})['user_notes'] = []

            # Append note with timestamp
            conv['metadata']['user_notes'].append({
                'note': note,
                'added_at': datetime.now().isoformat()
            })

            # Save
            with open(conv_file, 'w', encoding='utf-8') as f:
                json.dump(conv, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"Error adding user note: {e}")
            return False

    def set_conversation_relevance(self, conversation_id: str, relevance: str) -> bool:
        """
        Manually set conversation relevance level.

        Args:
            conversation_id: ID of conversation
            relevance: 'critical', 'high', 'medium', 'low', or 'archive'

        Returns:
            Success boolean
        """
        conv_file = self.conversations_path / f"{conversation_id}.json"

        if not conv_file.exists():
            return False

        try:
            with open(conv_file, 'r', encoding='utf-8') as f:
                conv = json.load(f)

            # Set manual relevance (overrides auto-importance)
            conv.setdefault('metadata', {})['manual_relevance'] = relevance
            conv['metadata']['relevance_set_at'] = datetime.now().isoformat()

            # Save
            with open(conv_file, 'w', encoding='utf-8') as f:
                json.dump(conv, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"Error setting relevance: {e}")
            return False


# Example usage
if __name__ == "__main__":
    memory = UltimateMemoryService()

    # Test conversation
    test_messages = [
        {
            "role": "user",
            "content": "I want to set up auto-save for all my conversations to my NAS storage"
        },
        {
            "role": "assistant",
            "content": "I'll configure the SessionEnd hook to auto-save conversations"
        }
    ]

    conv_id = memory.save_conversation(
        messages=test_messages,
        metadata={"test": True}
    )

    print(f"Saved conversation: {conv_id}")
