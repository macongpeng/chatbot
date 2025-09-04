#!/usr/bin/env python3
"""
Test Comprehensive Baseline - Clean 7-Query Test Suite
Establish true baseline performance with comprehensive query set and no enhancements
"""

import json
import statistics
import time
from typing import List, Dict, Any

def test_comprehensive_baseline():
    """Test comprehensive baseline with clean implementation and 7-query test suite"""
    
    print("🚀 TESTING COMPREHENSIVE BASELINE - CLEAN IMPLEMENTATION")
    print("=" * 65)
    
    # COMPREHENSIVE 7-query test suite (our standard test set)
    test_queries = [
        {
            "query": "How do I create an appointment in MediRecords?",
            "expected_keywords": ["appointment", "create", "schedule", "booking"],
            "difficulty": "easy",
            "category": "appointments"
        },
        {
            "query": "Steps to process Medicare billing claims?", 
            "expected_keywords": ["medicare", "billing", "claim", "payment"],
            "difficulty": "easy",
            "category": "billing"
        },
        {
            "query": "Where is the patient registration located?",
            "expected_keywords": ["patient", "register", "location", "find"],
            "difficulty": "medium",
            "category": "patients"
        },
        {
            "query": "EMR system not working properly",
            "expected_keywords": ["system", "error", "problem", "fix"],
            "difficulty": "hard",
            "category": "troubleshooting"
        },
        {
            "query": "What are the clinical features in MediRecord?",
            "expected_keywords": ["clinical", "features", "medical", "capabilities"],
            "difficulty": "medium",
            "category": "clinical"
        },
        {
            "query": "Setup SMS reminders for appointments",
            "expected_keywords": ["sms", "reminder", "appointment", "setup"],
            "difficulty": "medium", 
            "category": "reminders"
        },
        {
            "query": "Generate billing reports for patients", 
            "expected_keywords": ["report", "billing", "generate", "patient"],
            "difficulty": "medium",
            "category": "reports"
        }
    ]
    
    print(f"🔍 Testing with COMPREHENSIVE {len(test_queries)}-query baseline test suite...")
    print("🎯 This will be our new performance benchmark for all future Phase 2 work")
    
    try:
        from medibotllamaindex import hybrid_retriever
        
        query_results = []
        difficulty_results = {"easy": [], "medium": [], "hard": []}
        
        for i, query_info in enumerate(test_queries, 1):
            query = query_info["query"]
            expected_keywords = query_info["expected_keywords"]
            difficulty = query_info["difficulty"]
            category = query_info["category"]
            
            print(f"\\n   🔍 Query {i}/7: {query}")
            print(f"      Category: {category} | Difficulty: {difficulty}")
            
            try:
                # Test retrieval performance using clean baseline
                start_time = time.time()
                nodes = hybrid_retriever.retrieve(query)
                retrieval_time = time.time() - start_time
                
                relevance_scores = []
                high_relevance_count = 0
                
                for node in nodes:
                    content = node.node.text.lower()
                    
                    # Standard relevance calculation
                    matches = [kw for kw in expected_keywords if kw.lower() in content]
                    relevance_score = len(matches) / len(expected_keywords)
                    relevance_scores.append(relevance_score)
                    
                    if relevance_score >= 0.5:
                        high_relevance_count += 1
                
                precision = high_relevance_count / len(nodes) if nodes else 0.0
                avg_relevance = statistics.mean(relevance_scores) if relevance_scores else 0.0
                
                # Track by difficulty
                difficulty_results[difficulty].append(precision)
                
                query_result = {
                    'query': query,
                    'category': category,
                    'difficulty': difficulty,
                    'precision': precision,
                    'avg_relevance': avg_relevance,
                    'retrieval_time': retrieval_time,
                    'nodes_count': len(nodes)
                }
                query_results.append(query_result)
                
                print(f"      📊 P={precision:.3f} ({precision*100:.1f}%), R={avg_relevance:.3f} ({avg_relevance*100:.1f}%)")
                print(f"      📄 Retrieved {len(nodes)} nodes in {retrieval_time:.3f}s")
                
            except Exception as e:
                print(f"      ❌ Query failed: {e}")
                continue
        
        if query_results:
            # Calculate comprehensive baseline metrics
            avg_precision = statistics.mean([r['precision'] for r in query_results])
            avg_relevance = statistics.mean([r['avg_relevance'] for r in query_results])
            avg_time = statistics.mean([r['retrieval_time'] for r in query_results])
            
            # Calculate by difficulty
            easy_precision = statistics.mean(difficulty_results["easy"]) if difficulty_results["easy"] else 0.0
            medium_precision = statistics.mean(difficulty_results["medium"]) if difficulty_results["medium"] else 0.0
            hard_precision = statistics.mean(difficulty_results["hard"]) if difficulty_results["hard"] else 0.0
            
            print(f"\\n🏆 COMPREHENSIVE BASELINE RESULTS:")
            print(f"   📈 Overall Precision: {avg_precision:.4f} ({avg_precision*100:.1f}%)")
            print(f"   📈 Overall Relevance: {avg_relevance:.4f} ({avg_relevance*100:.1f}%)")
            print(f"   ⚡ Average Speed: {avg_time:.3f}s")
            
            print(f"\\n📊 PERFORMANCE BY DIFFICULTY:")
            print(f"   🟢 Easy Queries:   {easy_precision:.3f} ({easy_precision*100:.1f}%) - {len(difficulty_results['easy'])} queries")
            print(f"   🟡 Medium Queries: {medium_precision:.3f} ({medium_precision*100:.1f}%) - {len(difficulty_results['medium'])} queries")
            print(f"   🔴 Hard Queries:   {hard_precision:.3f} ({hard_precision*100:.1f}%) - {len(difficulty_results['hard'])} queries")
            
            print(f"\\n🎯 NEW BASELINE ESTABLISHED:")
            print(f"   📍 Precision Baseline: {avg_precision*100:.1f}%")
            print(f"   📍 Relevance Baseline: {avg_relevance*100:.1f}%")
            print(f"   📍 This replaces the optimistic 91.7% from limited 3-query validation")
            print(f"   📍 All future Phase 2 improvements will be measured against this baseline")
            
            # Performance assessment
            if avg_precision >= 0.90:
                performance_grade = "EXCELLENT"
            elif avg_precision >= 0.80:
                performance_grade = "GOOD" 
            elif avg_precision >= 0.70:
                performance_grade = "FAIR"
            else:
                performance_grade = "NEEDS IMPROVEMENT"
            
            print(f"\\n   📝 Baseline Grade: {performance_grade}")
            print(f"   📈 Target for Phase 2: >{avg_precision*100 + 3:.1f}% precision (+3% improvement)")
            
            return {
                'avg_precision': avg_precision,
                'avg_relevance': avg_relevance,
                'avg_time': avg_time,
                'easy_precision': easy_precision,
                'medium_precision': medium_precision,
                'hard_precision': hard_precision,
                'performance_grade': performance_grade,
                'query_results': query_results
            }
            
    except Exception as e:
        print(f"   ❌ Comprehensive baseline test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = test_comprehensive_baseline()
    
    if results:
        with open('tests/comprehensive_baseline_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\n💾 Results saved to 'tests/comprehensive_baseline_results.json'")
        print("🎯 Comprehensive baseline testing complete!")
        print("✅ New baseline established for Phase 2 development!")
        print("🚀 Ready to proceed with Phase 2 enhancements against accurate benchmark!")
    else:
        print("❌ Comprehensive baseline testing failed!")