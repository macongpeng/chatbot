#!/usr/bin/env python3
"""
Test Phase 2B - Enhanced Query Processing 
Measure performance improvements from intent detection and medical term expansion
"""

import json
import statistics
import time
from typing import List, Dict, Any

def test_phase2b_improvements():
    """Test Phase 2B enhanced query processing improvements"""
    
    print("🚀 TESTING PHASE 2B - ENHANCED QUERY PROCESSING")
    print("=" * 60)
    
    # Diverse test queries to showcase intent detection and term expansion
    test_queries = [
        {
            "query": "How do I create an appt in MediRecords?",
            "expected_intent": "procedure",
            "expected_category": "appointments",
            "expected_keywords": ["appointment", "create", "schedule", "booking"],
            "contains_abbreviation": True
        },
        {
            "query": "Steps to process Medicare billing claims?", 
            "expected_intent": "procedure",
            "expected_category": "billing",
            "expected_keywords": ["medicare", "billing", "claim", "payment"],
            "contains_abbreviation": False
        },
        {
            "query": "Where is the pt registration located?",
            "expected_intent": "reference", 
            "expected_category": "patients",
            "expected_keywords": ["patient", "register", "location", "find"],
            "contains_abbreviation": True
        },
        {
            "query": "EMR system not working properly",
            "expected_intent": "troubleshooting",
            "expected_category": "system", 
            "expected_keywords": ["system", "error", "problem", "fix"],
            "contains_abbreviation": True
        },
        {
            "query": "What are the clinical features in MediRecord?",
            "expected_intent": "overview",
            "expected_category": "clinical",
            "expected_keywords": ["clinical", "features", "medical", "capabilities"],
            "contains_abbreviation": False
        },
        {
            "query": "Setup SMS reminders for appointments",
            "expected_intent": "procedure",
            "expected_category": "reminders",
            "expected_keywords": ["sms", "reminder", "appointment", "setup"],
            "contains_abbreviation": False
        },
        {
            "query": "Generate billing reports for pts", 
            "expected_intent": "procedure",
            "expected_category": "reports",
            "expected_keywords": ["report", "billing", "generate", "patient"],
            "contains_abbreviation": True
        }
    ]
    
    print("🔍 Testing Phase 2B Enhanced Query Processing...")
    
    try:
        from medibotllamaindex import (hybrid_retriever, detect_query_intent, 
                                       detect_query_category, preprocess_medical_query,
                                       expand_medical_terms)
        
        query_results = []
        intent_accuracy = []
        category_accuracy = []
        abbreviation_expansion_success = []
        
        for query_info in test_queries:
            query = query_info["query"]
            expected_intent = query_info["expected_intent"]
            expected_category = query_info["expected_category"] 
            expected_keywords = query_info["expected_keywords"]
            has_abbreviation = query_info["contains_abbreviation"]
            
            print(f"\\n   🔍 Testing: {query}")
            print(f"      Expected: {expected_intent}/{expected_category}")
            
            try:
                # Test Phase 2B components individually
                detected_intent = detect_query_intent(query)
                detected_category = detect_query_category(query)
                expanded_query = expand_medical_terms(query)
                processed_query = preprocess_medical_query(query)
                
                # Check intent and category accuracy
                intent_correct = detected_intent == expected_intent
                category_correct = detected_category == expected_category
                
                intent_accuracy.append(1.0 if intent_correct else 0.0)
                category_accuracy.append(1.0 if category_correct else 0.0)
                
                # Check abbreviation expansion
                if has_abbreviation:
                    abbreviation_expanded = len(expanded_query) > len(query)
                    abbreviation_expansion_success.append(1.0 if abbreviation_expanded else 0.0)
                
                print(f"      🎯 Detected: {detected_intent}/{detected_category} {'✅' if intent_correct and category_correct else '⚠️'}")
                if has_abbreviation:
                    print(f"      📝 Expansion: {query} → {expanded_query[:60]}...")
                
                # Test retrieval performance 
                start_time = time.time()
                nodes = hybrid_retriever.retrieve(processed_query)
                retrieval_time = time.time() - start_time
                
                relevance_scores = []
                high_relevance_count = 0
                quality_scores = []
                
                for node in nodes:
                    content = node.node.text.lower()
                    metadata = node.node.metadata
                    
                    # Standard relevance calculation
                    matches = [kw for kw in expected_keywords if kw.lower() in content]
                    relevance_score = len(matches) / len(expected_keywords)
                    relevance_scores.append(relevance_score)
                    
                    quality_score = metadata.get('quality_score', 0.5)
                    quality_scores.append(quality_score)
                    
                    if relevance_score >= 0.5:
                        high_relevance_count += 1
                
                precision = high_relevance_count / len(nodes) if nodes else 0.0
                avg_relevance = statistics.mean(relevance_scores) if relevance_scores else 0.0
                avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0
                
                query_results.append({
                    'query': query,
                    'intent_correct': intent_correct,
                    'category_correct': category_correct,
                    'precision': precision,
                    'avg_relevance': avg_relevance,
                    'avg_quality': avg_quality,
                    'retrieval_time': retrieval_time
                })
                
                print(f"      📊 P={precision:.3f}, R={avg_relevance:.3f}, Q={avg_quality:.3f}")
                
            except Exception as e:
                print(f"      ❌ Query failed: {e}")
                continue
        
        if query_results:
            # Calculate overall metrics
            avg_precision = statistics.mean([r['precision'] for r in query_results])
            avg_relevance = statistics.mean([r['avg_relevance'] for r in query_results])
            avg_quality = statistics.mean([r['avg_quality'] for r in query_results])
            avg_intent_accuracy = statistics.mean(intent_accuracy)
            avg_category_accuracy = statistics.mean(category_accuracy)
            avg_abbreviation_expansion = statistics.mean(abbreviation_expansion_success) if abbreviation_expansion_success else 0.0
            avg_time = statistics.mean([r['retrieval_time'] for r in query_results])
            
            print(f"\\n🏆 PHASE 2B ENHANCED RESULTS:")
            print(f"   📈 Precision: {avg_precision:.3f} ({avg_precision*100:.1f}%)")
            print(f"   📈 Relevance: {avg_relevance:.3f} ({avg_relevance*100:.1f}%)")
            print(f"   📈 Content Quality: {avg_quality:.3f} ({avg_quality*100:.1f}%)")
            print(f"   🎯 Intent Detection: {avg_intent_accuracy:.3f} ({avg_intent_accuracy*100:.1f}% accuracy)")
            print(f"   🏷️  Category Detection: {avg_category_accuracy:.3f} ({avg_category_accuracy*100:.1f}% accuracy)")
            print(f"   📝 Abbreviation Expansion: {avg_abbreviation_expansion:.3f} ({avg_abbreviation_expansion*100:.1f}% success)")
            print(f"   ⚡ Speed: {avg_time:.3f}s")
            
            # Compare with baseline (Phase 2A results)
            baseline_precision = 91.7  # Original champion performance
            baseline_relevance = 76.7
            
            precision_improvement = (avg_precision * 100) - baseline_precision
            relevance_improvement = (avg_relevance * 100) - baseline_relevance
            
            print(f"\\n📊 PHASE 2B IMPROVEMENT ANALYSIS:")
            print(f"   🔥 Precision Change: {precision_improvement:+.1f}% ({baseline_precision:.1f}% → {avg_precision*100:.1f}%)")
            print(f"   🔥 Relevance Change: {relevance_improvement:+.1f}% ({baseline_relevance:.1f}% → {avg_relevance*100:.1f}%)")
            
            print(f"\\n🎯 PHASE 2B SPECIFIC ACHIEVEMENTS:")
            if avg_intent_accuracy >= 0.8:
                print(f"   ✅ EXCELLENT intent detection system")
            if avg_category_accuracy >= 0.7:
                print(f"   ✅ STRONG category classification")
            if avg_abbreviation_expansion >= 0.8:
                print(f"   ✅ EFFECTIVE medical term expansion")
            
            # Overall Phase 2B assessment
            if precision_improvement >= 3.0:
                print(f"\\n   🎉 OUTSTANDING: Exceeded Phase 2B targets significantly!")
            elif precision_improvement >= 1.0:
                print(f"\\n   🚀 EXCELLENT: Phase 2B delivered strong improvements!")
            elif precision_improvement >= 0.0:
                print(f"\\n   👍 POSITIVE: Phase 2B showing measurable benefits!")
            else:
                print(f"\\n   🔧 ANALYSIS: Phase 2B components working, optimization needed")
            
            return {
                'avg_precision': avg_precision,
                'avg_relevance': avg_relevance,
                'avg_quality': avg_quality,
                'avg_intent_accuracy': avg_intent_accuracy,
                'avg_category_accuracy': avg_category_accuracy,
                'avg_abbreviation_expansion': avg_abbreviation_expansion,
                'avg_time': avg_time,
                'precision_improvement': precision_improvement,
                'relevance_improvement': relevance_improvement,
                'query_results': query_results
            }
            
    except Exception as e:
        print(f"   ❌ Phase 2B test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = test_phase2b_improvements()
    
    if results:
        with open('tests/phase2b_improvement_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\n💾 Results saved to 'tests/phase2b_improvement_results.json'")
        print("🎯 Phase 2B testing complete!")
        print("🚀 Ready for final Phase 2 validation and Phase 2C if needed!")
    else:
        print("❌ Phase 2B testing failed!")