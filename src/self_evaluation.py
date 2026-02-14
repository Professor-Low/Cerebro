"""
Self Evaluation - Generate periodic self-improvement reports.

Part of Phase 7: Feedback Loops in the Brain Evolution Plan.

Generates:
- Correction rate trends (how often Claude is corrected)
- Solution success rate (solutions that worked vs failed)
- Fact decay rate (how many facts are becoming stale)
- Memory growth rate (storage usage trends)
- Pattern promotion candidates (recurring issues)
- Comprehensive self-improvement reports
"""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ImprovementRecord:
    """Record of a self-improvement or learning."""
    improvement_id: str
    description: str
    baseline_value: float
    current_value: float
    improvement_pct: float
    timestamp: str
    category: str  # 'accuracy', 'efficiency', 'knowledge', 'response_quality'


class SelfEvaluator:
    """
    Generate self-evaluation reports and track improvement over time.

    This is Claude's introspection system - tracking how well
    responses are being received and where improvement is needed.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            from config import AI_MEMORY_BASE
            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self.reports_path = self.base_path / "self_reports"
        self.reports_path.mkdir(parents=True, exist_ok=True)

        # Paths to data sources
        self.corrections_path = self.base_path / "corrections"
        self.conversations_path = self.base_path / "conversations"
        self.facts_path = self.base_path / "facts"
        self.solutions_path = self.base_path / "solutions"
        self.feedback_path = self.base_path / "feedback"
        self.metrics_path = self.base_path / "metrics"

    def calculate_correction_rate(self, days: int = 30) -> Dict[str, Any]:
        """
        Calculate the correction rate over time.

        Correction rate = number of corrections / number of conversations
        Lower is better.
        """
        cutoff = datetime.now() - timedelta(days=days)

        # Count corrections
        corrections_file = self.corrections_path / "corrections.json"
        corrections_count = 0
        corrections_by_day = defaultdict(int)

        if corrections_file.exists():
            try:
                with open(corrections_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for corr in data.get('corrections', []):
                    ts = corr.get('timestamp', '')
                    if ts >= cutoff.isoformat():
                        corrections_count += 1
                        day = ts[:10]
                        corrections_by_day[day] += 1
            except Exception:
                pass

        # Count conversations
        conversations_count = 0
        conversations_by_day = defaultdict(int)

        if self.conversations_path.exists():
            for conv_file in self.conversations_path.glob("*.json"):
                try:
                    with open(conv_file, 'r', encoding='utf-8') as f:
                        conv = json.load(f)

                    ts = conv.get('timestamp', conv.get('created_at', ''))
                    if ts >= cutoff.isoformat():
                        conversations_count += 1
                        day = ts[:10]
                        conversations_by_day[day] += 1
                except Exception:
                    continue

        # Calculate rate
        rate = corrections_count / max(conversations_count, 1)

        # Calculate trend
        trend = self._calculate_trend(corrections_by_day, conversations_by_day)

        return {
            'period_days': days,
            'total_corrections': corrections_count,
            'total_conversations': conversations_count,
            'correction_rate': round(rate * 100, 2),  # As percentage
            'rate_unit': 'percent',
            'interpretation': 'Lower is better',
            'trend': trend,
            'daily_corrections': dict(corrections_by_day),
            'daily_conversations': dict(conversations_by_day)
        }

    def calculate_solution_success_rate(self, days: int = 30) -> Dict[str, Any]:
        """
        Calculate solution success rate from feedback and solution data.

        Success rate = confirmed solutions / (confirmed + failed)
        Higher is better.
        """
        cutoff = datetime.now() - timedelta(days=days)

        total_solutions = 0
        confirmed = 0
        failed = 0
        by_day = defaultdict(lambda: {'confirmed': 0, 'failed': 0})

        if self.solutions_path.exists():
            for sol_file in self.solutions_path.glob("*.json"):
                try:
                    with open(sol_file, 'r', encoding='utf-8') as f:
                        sol = json.load(f)

                    # Check if within time period
                    ts = sol.get('updated_at', sol.get('created_at', ''))
                    if ts >= cutoff.isoformat():
                        total_solutions += 1
                        day = ts[:10]

                        success_count = sol.get('success_confirmations', 0)
                        failure_count = sol.get('failure_count', 0)

                        if success_count > 0:
                            confirmed += 1
                            by_day[day]['confirmed'] += 1
                        if failure_count > 0:
                            failed += 1
                            by_day[day]['failed'] += 1

                except Exception:
                    continue

        # Also check feedback log
        feedback_log = self.feedback_path / "feedback_log.jsonl"
        if feedback_log.exists():
            try:
                with open(feedback_log, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            record = json.loads(line.strip())
                            ts = record.get('timestamp', '')
                            if ts >= cutoff.isoformat():
                                day = ts[:10]
                                if record.get('signal_type') == 'success':
                                    by_day[day]['confirmed'] += 1
                                    confirmed += 1
                                elif record.get('signal_type') == 'failure':
                                    by_day[day]['failed'] += 1
                                    failed += 1
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass

        total_feedback = confirmed + failed
        success_rate = confirmed / max(total_feedback, 1)

        return {
            'period_days': days,
            'total_solutions_tracked': total_solutions,
            'confirmed_working': confirmed,
            'confirmed_failed': failed,
            'success_rate': round(success_rate * 100, 2),
            'rate_unit': 'percent',
            'interpretation': 'Higher is better',
            'by_day': dict(by_day)
        }

    def calculate_fact_health(self) -> Dict[str, Any]:
        """
        Calculate fact health metrics.

        Tracks:
        - Active vs superseded facts
        - Low confidence facts
        - Fact age distribution
        """
        if not self.facts_path.exists():
            return {'error': 'Facts path not found'}

        total = 0
        active = 0
        superseded = 0
        low_confidence = 0
        age_distribution = {'<30_days': 0, '30-90_days': 0, '90-180_days': 0, '>180_days': 0}
        confidence_distribution = {'high': 0, 'medium': 0, 'low': 0}

        now = datetime.now()

        for fact_file in self.facts_path.glob("*.json"):
            try:
                with open(fact_file, 'r', encoding='utf-8') as f:
                    fact = json.load(f)

                total += 1

                # Status
                status = fact.get('status', fact.get('superseded'))
                if status == 'superseded' or status is True:
                    superseded += 1
                else:
                    active += 1

                # Confidence
                conf = fact.get('confidence', 0.7)
                if isinstance(conf, str):
                    conf = {'high': 0.9, 'medium': 0.7, 'low': 0.4}.get(conf, 0.5)

                if conf >= 0.7:
                    confidence_distribution['high'] += 1
                elif conf >= 0.4:
                    confidence_distribution['medium'] += 1
                else:
                    confidence_distribution['low'] += 1
                    low_confidence += 1

                # Age
                created = fact.get('created_at', fact.get('timestamp', ''))
                if created:
                    try:
                        created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                        age_days = (now - created_dt.replace(tzinfo=None)).days

                        if age_days < 30:
                            age_distribution['<30_days'] += 1
                        elif age_days < 90:
                            age_distribution['30-90_days'] += 1
                        elif age_days < 180:
                            age_distribution['90-180_days'] += 1
                        else:
                            age_distribution['>180_days'] += 1
                    except Exception:
                        pass

            except Exception:
                continue

        health_score = 1.0
        # Penalize for high supersession rate
        if total > 0:
            supersession_rate = superseded / total
            health_score -= supersession_rate * 0.3

            # Penalize for high low-confidence rate
            low_conf_rate = low_confidence / total
            health_score -= low_conf_rate * 0.2

        return {
            'total_facts': total,
            'active_facts': active,
            'superseded_facts': superseded,
            'supersession_rate': round(superseded / max(total, 1) * 100, 2),
            'low_confidence_facts': low_confidence,
            'low_confidence_rate': round(low_confidence / max(total, 1) * 100, 2),
            'confidence_distribution': confidence_distribution,
            'age_distribution': age_distribution,
            'health_score': round(max(0, health_score), 2)
        }

    def calculate_memory_growth(self, days: int = 30) -> Dict[str, Any]:
        """
        Calculate memory growth rate.

        Tracks storage usage over time.
        """
        directories = {
            'conversations': self.conversations_path,
            'facts': self.facts_path,
            'solutions': self.solutions_path,
            'feedback': self.feedback_path
        }

        growth = {}
        total_size = 0
        total_files = 0

        for name, path in directories.items():
            if path.exists():
                files = list(path.rglob('*'))
                file_count = len([f for f in files if f.is_file()])
                dir_size = sum(f.stat().st_size for f in files if f.is_file())

                growth[name] = {
                    'files': file_count,
                    'size_bytes': dir_size,
                    'size_mb': round(dir_size / (1024 * 1024), 2)
                }

                total_size += dir_size
                total_files += file_count

        # Calculate growth rate from conversation timestamps
        recent_count = 0
        older_count = 0
        midpoint = datetime.now() - timedelta(days=days // 2)
        cutoff = datetime.now() - timedelta(days=days)

        if self.conversations_path.exists():
            for conv_file in self.conversations_path.glob("*.json"):
                try:
                    ts = datetime.fromtimestamp(conv_file.stat().st_mtime)
                    if ts >= midpoint:
                        recent_count += 1
                    elif ts >= cutoff:
                        older_count += 1
                except Exception:
                    continue

        # Growth rate: recent half vs older half
        if older_count > 0:
            growth_rate = (recent_count - older_count) / older_count
        else:
            growth_rate = 1.0 if recent_count > 0 else 0.0

        return {
            'period_days': days,
            'total_files': total_files,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'by_directory': growth,
            'growth_rate': round(growth_rate * 100, 2),
            'growth_interpretation': 'positive means growing' if growth_rate >= 0 else 'negative means shrinking',
            'recent_half_conversations': recent_count,
            'older_half_conversations': older_count
        }

    def find_pattern_promotion_candidates(self, min_occurrences: int = 3) -> Dict[str, Any]:
        """
        Find recurring problems that should be promoted to permanent context.

        When a problem/solution pair occurs multiple times, it should be
        added to quick_facts or permanent context.
        """
        problem_occurrences = defaultdict(list)
        solution_effectiveness = defaultdict(lambda: {'success': 0, 'failure': 0})

        # Scan solutions for recurring problems
        if self.solutions_path.exists():
            for sol_file in self.solutions_path.glob("*.json"):
                try:
                    with open(sol_file, 'r', encoding='utf-8') as f:
                        sol = json.load(f)

                    problem_hash = sol.get('problem_hash', '')
                    problem_text = sol.get('problem', '')

                    if problem_hash:
                        problem_occurrences[problem_hash].append({
                            'problem': problem_text[:200],
                            'solution': sol.get('solution', '')[:200],
                            'solution_id': sol.get('id'),
                            'confirmations': sol.get('success_confirmations', 0),
                            'failures': sol.get('failure_count', 0)
                        })

                        # Track effectiveness
                        solution_effectiveness[problem_hash]['success'] += sol.get('success_confirmations', 0)
                        solution_effectiveness[problem_hash]['failure'] += sol.get('failure_count', 0)

                except Exception:
                    continue

        # Find candidates with enough occurrences
        candidates = []
        for problem_hash, occurrences in problem_occurrences.items():
            if len(occurrences) >= min_occurrences:
                effectiveness = solution_effectiveness[problem_hash]
                total = effectiveness['success'] + effectiveness['failure']
                success_rate = effectiveness['success'] / max(total, 1)

                # Only promote if solution has reasonable success rate
                if success_rate >= 0.6 or total == 0:  # 60% success or no feedback yet
                    candidates.append({
                        'problem_hash': problem_hash,
                        'problem_text': occurrences[0]['problem'],
                        'best_solution': occurrences[0]['solution'],
                        'occurrence_count': len(occurrences),
                        'total_confirmations': effectiveness['success'],
                        'total_failures': effectiveness['failure'],
                        'success_rate': round(success_rate * 100, 2) if total > 0 else None,
                        'recommendation': 'promote_to_quick_facts' if success_rate >= 0.8 else 'monitor'
                    })

        # Sort by occurrence count
        candidates.sort(key=lambda x: x['occurrence_count'], reverse=True)

        return {
            'min_occurrences_threshold': min_occurrences,
            'candidates_found': len(candidates),
            'candidates': candidates[:20],  # Top 20
            'action_items': [c for c in candidates if c['recommendation'] == 'promote_to_quick_facts']
        }

    def promote_pattern_to_quick_facts(self,
                                        problem_hash: str,
                                        problem_text: str,
                                        solution_text: str,
                                        source: str = "pattern_promotion") -> Dict[str, Any]:
        """
        Promote a recurring pattern to quick_facts.json for permanent context.

        This adds the problem/solution to the 'promoted_patterns' section
        of quick_facts.json so Claude always has access to it.

        Args:
            problem_hash: Hash of the problem for deduplication
            problem_text: Description of the problem
            solution_text: The proven solution
            source: Source of the promotion (default: pattern_promotion)

        Returns:
            Result of the promotion
        """
        quick_facts_path = self.base_path / "quick_facts.json"

        try:
            # Load existing quick_facts
            if quick_facts_path.exists():
                with open(quick_facts_path, 'r', encoding='utf-8') as f:
                    quick_facts = json.load(f)
            else:
                quick_facts = {}

            # Initialize promoted_patterns if needed
            if 'promoted_patterns' not in quick_facts:
                quick_facts['promoted_patterns'] = {
                    '_description': 'Recurring problems and their proven solutions, auto-promoted from learning system',
                    'patterns': []
                }

            # Check if already promoted
            existing_hashes = {
                p.get('problem_hash') for p in quick_facts['promoted_patterns']['patterns']
            }
            if problem_hash in existing_hashes:
                return {
                    'success': False,
                    'reason': 'Pattern already promoted',
                    'problem_hash': problem_hash
                }

            # Add the pattern
            pattern = {
                'problem_hash': problem_hash,
                'problem': problem_text[:300],
                'solution': solution_text[:500],
                'promoted_at': datetime.now().isoformat(),
                'source': source
            }
            quick_facts['promoted_patterns']['patterns'].append(pattern)

            # Keep only last 50 patterns to avoid bloat
            if len(quick_facts['promoted_patterns']['patterns']) > 50:
                quick_facts['promoted_patterns']['patterns'] = \
                    quick_facts['promoted_patterns']['patterns'][-50:]

            # Save
            with open(quick_facts_path, 'w', encoding='utf-8') as f:
                json.dump(quick_facts, f, indent=2, ensure_ascii=False)

            return {
                'success': True,
                'problem_hash': problem_hash,
                'problem': problem_text[:100],
                'message': 'Pattern promoted to quick_facts.json'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'problem_hash': problem_hash
            }

    def auto_promote_patterns(self, min_success_rate: float = 0.8) -> Dict[str, Any]:
        """
        Automatically promote all qualifying patterns to quick_facts.

        Args:
            min_success_rate: Minimum success rate to auto-promote (default 80%)

        Returns:
            Summary of promotions
        """
        candidates = self.find_pattern_promotion_candidates()
        action_items = candidates.get('action_items', [])

        promoted = []
        skipped = []

        for candidate in action_items:
            if candidate.get('success_rate', 0) >= min_success_rate * 100:
                result = self.promote_pattern_to_quick_facts(
                    problem_hash=candidate['problem_hash'],
                    problem_text=candidate['problem_text'],
                    solution_text=candidate['best_solution']
                )
                if result.get('success'):
                    promoted.append(candidate['problem_text'][:50])
                else:
                    skipped.append({
                        'problem': candidate['problem_text'][:50],
                        'reason': result.get('reason', 'unknown')
                    })

        return {
            'total_candidates': len(action_items),
            'promoted_count': len(promoted),
            'skipped_count': len(skipped),
            'promoted': promoted,
            'skipped': skipped
        }

    def generate_full_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate a comprehensive self-evaluation report.

        This is the main report that combines all metrics and provides
        actionable insights for self-improvement.
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'period_days': days,
            'report_type': 'comprehensive_self_evaluation',

            # Core metrics
            'correction_rate': self.calculate_correction_rate(days),
            'solution_success': self.calculate_solution_success_rate(days),
            'fact_health': self.calculate_fact_health(),
            'memory_growth': self.calculate_memory_growth(days),

            # Pattern analysis
            'pattern_promotion': self.find_pattern_promotion_candidates(),

            # Summary scores
            'summary': {}
        }

        # Calculate summary scores
        scores = []

        # Correction rate score (lower is better, 0-5% is excellent)
        corr_rate = report['correction_rate'].get('correction_rate', 0)
        corr_score = max(0, 1 - (corr_rate / 20))  # 20% = 0, 0% = 1
        scores.append(('accuracy', corr_score))

        # Solution success score
        success_rate = report['solution_success'].get('success_rate', 50) / 100
        scores.append(('solution_effectiveness', success_rate))

        # Fact health score
        fact_health = report['fact_health'].get('health_score', 0.5)
        scores.append(('knowledge_quality', fact_health))

        # Overall score
        overall = sum(s[1] for s in scores) / len(scores) if scores else 0.5

        report['summary'] = {
            'overall_score': round(overall, 2),
            'component_scores': {name: round(score, 2) for name, score in scores},
            'grade': self._score_to_grade(overall),
            'areas_for_improvement': self._identify_improvements(report),
            'strengths': self._identify_strengths(report)
        }

        # Save report
        self._save_report(report)

        return report

    def _calculate_trend(self, numerator_by_day: Dict, denominator_by_day: Dict) -> str:
        """Calculate trend direction from daily data."""
        if not numerator_by_day or not denominator_by_day:
            return 'insufficient_data'

        # Get last 7 days vs previous 7 days
        days = sorted(set(numerator_by_day.keys()) | set(denominator_by_day.keys()))

        if len(days) < 7:
            return 'insufficient_data'

        recent_days = days[-7:]
        older_days = days[-14:-7] if len(days) >= 14 else days[:7]

        recent_rate = sum(numerator_by_day.get(d, 0) for d in recent_days) / \
                      max(sum(denominator_by_day.get(d, 0) for d in recent_days), 1)

        older_rate = sum(numerator_by_day.get(d, 0) for d in older_days) / \
                     max(sum(denominator_by_day.get(d, 0) for d in older_days), 1)

        diff = recent_rate - older_rate

        if abs(diff) < 0.01:
            return 'stable'
        elif diff > 0:
            return 'increasing'  # For correction rate, this is bad
        else:
            return 'decreasing'  # For correction rate, this is good

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'

    def _identify_improvements(self, report: Dict) -> List[str]:
        """Identify areas needing improvement from report data."""
        improvements = []

        # High correction rate
        corr_rate = report.get('correction_rate', {}).get('correction_rate', 0)
        if corr_rate > 10:
            improvements.append(f"Correction rate is {corr_rate}% - consider reviewing common correction patterns")

        # Low solution success
        success_rate = report.get('solution_success', {}).get('success_rate', 100)
        if success_rate < 70:
            improvements.append(f"Solution success rate is {success_rate}% - review failed solutions for patterns")

        # Many low-confidence facts
        low_conf = report.get('fact_health', {}).get('low_confidence_rate', 0)
        if low_conf > 20:
            improvements.append(f"{low_conf}% of facts have low confidence - consider verification")

        # High supersession rate
        super_rate = report.get('fact_health', {}).get('supersession_rate', 0)
        if super_rate > 5:
            improvements.append(f"Fact supersession rate is {super_rate}% - knowledge is changing frequently")

        # Patterns to promote
        promotions = report.get('pattern_promotion', {}).get('action_items', [])
        if promotions:
            improvements.append(f"{len(promotions)} recurring patterns should be promoted to permanent context")

        return improvements if improvements else ["No significant improvement areas identified"]

    def _identify_strengths(self, report: Dict) -> List[str]:
        """Identify strengths from report data."""
        strengths = []

        # Low correction rate
        corr_rate = report.get('correction_rate', {}).get('correction_rate', 0)
        if corr_rate < 5:
            strengths.append(f"Excellent accuracy - only {corr_rate}% correction rate")

        # High solution success
        success_rate = report.get('solution_success', {}).get('success_rate', 0)
        if success_rate > 80:
            strengths.append(f"Strong solution effectiveness at {success_rate}%")

        # Good fact health
        health = report.get('fact_health', {}).get('health_score', 0)
        if health > 0.8:
            strengths.append(f"Knowledge base is healthy (score: {health})")

        # Many confirmed solutions
        confirmed = report.get('solution_success', {}).get('confirmed_working', 0)
        if confirmed > 10:
            strengths.append(f"{confirmed} solutions have been confirmed working")

        return strengths if strengths else ["Continuing to build knowledge base"]

    def _save_report(self, report: Dict) -> str:
        """Save report to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"self_report_{timestamp}.json"
        filepath = self.reports_path / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def get_recent_reports(self, limit: int = 5) -> List[Dict]:
        """Get most recent self-evaluation reports."""
        reports = []

        report_files = sorted(
            self.reports_path.glob("self_report_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:limit]

        for rf in report_files:
            try:
                with open(rf, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                report['_filename'] = rf.name
                reports.append(report)
            except Exception:
                continue

        return reports

    def track_improvement(self,
                          improvement_name: str,
                          description: str,
                          baseline_value: float,
                          current_value: float,
                          category: str = 'general') -> ImprovementRecord:
        """
        Record a specific improvement for tracking over time.
        """
        improvement_pct = ((current_value - baseline_value) / max(baseline_value, 0.01)) * 100

        record = ImprovementRecord(
            improvement_id=f"imp_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            description=description,
            baseline_value=baseline_value,
            current_value=current_value,
            improvement_pct=round(improvement_pct, 2),
            timestamp=datetime.now().isoformat(),
            category=category
        )

        # Save to improvements log
        improvements_file = self.reports_path / "improvements.jsonl"
        with open(improvements_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + '\n')

        return record

    def get_improvement_history(self, category: str = None) -> List[Dict]:
        """Get improvement records, optionally filtered by category."""
        improvements = []
        improvements_file = self.reports_path / "improvements.jsonl"

        if not improvements_file.exists():
            return improvements

        try:
            with open(improvements_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        if category is None or record.get('category') == category:
                            improvements.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

        return improvements


# Convenience functions
def generate_report(days: int = 30) -> Dict[str, Any]:
    """Generate a self-evaluation report."""
    evaluator = SelfEvaluator()
    return evaluator.generate_full_report(days)


def get_correction_rate(days: int = 30) -> Dict[str, Any]:
    """Get correction rate metrics."""
    evaluator = SelfEvaluator()
    return evaluator.calculate_correction_rate(days)


if __name__ == "__main__":
    evaluator = SelfEvaluator()

    print("=== Self Evaluation Report ===\n")

    # Generate full report
    report = evaluator.generate_full_report(days=30)

    print(f"Overall Score: {report['summary']['overall_score']} ({report['summary']['grade']})")
    print("\nComponent Scores:")
    for name, score in report['summary']['component_scores'].items():
        print(f"  - {name}: {score}")

    print(f"\nCorrection Rate: {report['correction_rate']['correction_rate']}%")
    print(f"Solution Success Rate: {report['solution_success']['success_rate']}%")
    print(f"Fact Health Score: {report['fact_health']['health_score']}")
    print(f"Total Facts: {report['fact_health']['total_facts']}")

    print("\nAreas for Improvement:")
    for area in report['summary']['areas_for_improvement']:
        print(f"  - {area}")

    print("\nStrengths:")
    for strength in report['summary']['strengths']:
        print(f"  - {strength}")

    print(f"\nPattern Promotion Candidates: {report['pattern_promotion']['candidates_found']}")
    for candidate in report['pattern_promotion']['candidates'][:3]:
        print(f"  - {candidate['problem_text'][:60]}... ({candidate['occurrence_count']}x)")
