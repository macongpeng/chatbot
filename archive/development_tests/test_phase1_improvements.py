#!/usr/bin/env python3
"""
Test Phase 1 Data Quality Improvements
Compare before/after performance metrics
"""

import json
import statistics
import time
from typing import List, Dict, Any

def test_phase1_improvements():
    """Test Phase 1 data quality improvements"""
    
    print("🚀 TESTING PHASE 1 DATA QUALITY IMPROVEMENTS")
    print("=" * 60)
    
    test_queries = [
        {
            "query": "How do I create an appointment in MediRecords?",
            "expected_keywords": ["appointment", "create", "schedule", "booking", "calendar"],
            "category": "appointment_management"
        },
        {
            "query": "What are the steps to process Medicare billing?", 
            "expected_keywords": ["medicare", "billing", "claim", "payment", "submit"],
            "category": "billing"
        },
        {
            "query": "How do I add a new patient to the system?",
            "expected_keywords": ["patient", "add", "new", "register", "create"],
            "category": "patient_management"
        },
        {
            "query": "How do I set up SMS appointment reminders?",
            "expected_keywords": ["sms", "reminder", "appointment", "notification", "setup"],
            "category": "reminders"
        },
        {
            "query": "Where do I find Medicare claiming options?",
            "expected_keywords": ["medicare", "claiming", "billing", "options", "find"],
            "category": "billing"
        }
    ]
    
    print("🔍 Testing Phase 1 Enhanced Champion Implementation...")
    
    try:
        from medibotllamaindex import hybrid_retriever, preprocess_medical_query
        
        query_results = []
        
        for query_info in test_queries:
            query = query_info["query"]
            expected_keywords = query_info["expected_keywords"]
            
            print(f"   🔍 Testing: {query}")
            
            try:
                start_time = time.time()
                processed_query = preprocess_medical_query(query)
                nodes = hybrid_retriever.retrieve(processed_query)
                retrieval_time = time.time() - start_time
                
                relevance_scores = []
                high_relevance_count = 0
                quality_scores = []
                
                for node in nodes:
                    content = node.node.text.lower()
                    matches = [kw for kw in expected_keywords if kw.lower() in content]
                    relevance_score = len(matches) / len(expected_keywords)
                    relevance_scores.append(relevance_score)
                    
                    # Get quality score from metadata
                    quality_score = node.node.metadata.get('quality_score', 0.5)
                    quality_scores.append(quality_score)
                    
                    if relevance_score >= 0.5:
                        high_relevance_count += 1
                
                precision = high_relevance_count / len(nodes) if nodes else 0.0
                avg_relevance = statistics.mean(relevance_scores) if relevance_scores else 0.0
                avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0
                
                query_results.append({
                    'query': query,
                    'precision': precision,
                    'avg_relevance': avg_relevance,
                    'avg_quality': avg_quality,
                    'retrieval_time': retrieval_time,
                    'nodes_count': len(nodes)
                })
                
                print(f"      📊 P={precision:.3f}, R={avg_relevance:.3f}, Q={avg_quality:.3f}, T={retrieval_time:.3f}s")
                
            except Exception as e:
                print(f"      ❌ Query failed: {e}")
                continue
        
        if query_results:
            avg_precision = statistics.mean([r['precision'] for r in query_results])
            avg_relevance = statistics.mean([r['avg_relevance'] for r in query_results])
            avg_quality = statistics.mean([r['avg_quality'] for r in query_results])
            avg_time = statistics.mean([r['retrieval_time'] for r in query_results])
            
            print(f"\\n🏆 PHASE 1 ENHANCED RESULTS:")
            print(f"   📈 Precision: {avg_precision:.3f} ({avg_precision*100:.1f}%)")
            print(f"   📈 Relevance: {avg_relevance:.3f} ({avg_relevance*100:.1f}%)")
            print(f"   📈 Content Quality: {avg_quality:.3f} ({avg_quality*100:.1f}%)")
            print(f"   ⚡ Speed: {avg_time:.3f}s")
            
            # Compare with baseline
            baseline_precision = 91.7
            baseline_relevance = 76.7
            
            precision_improvement = (avg_precision * 100) - baseline_precision
            relevance_improvement = (avg_relevance * 100) - baseline_relevance
            
            print(f"\\n📊 IMPROVEMENT ANALYSIS:")
            print(f"   🔥 Precision Change: {precision_improvement:+.1f}% ({baseline_precision:.1f}% → {avg_precision*100:.1f}%)")
            print(f"   🔥 Relevance Change: {relevance_improvement:+.1f}% ({baseline_relevance:.1f}% → {avg_relevance*100:.1f}%)")
            
            if precision_improvement >= 2.0:
                print(f"   🎉 EXCELLENT: Exceeded Phase 1 target (+2-3% precision)!")
            elif precision_improvement >= 1.0:
                print(f"   ✅ GOOD: Solid improvement in precision!")
            elif precision_improvement >= 0.0:
                print(f"   👍 POSITIVE: Some precision improvement achieved!")
            else:
                print(f"   ⚠️  CONCERN: Precision decreased - investigate further")
            
            return {
                'avg_precision': avg_precision,
                'avg_relevance': avg_relevance,
                'avg_quality': avg_quality,
                'avg_time': avg_time,
                'precision_improvement': precision_improvement,
                'relevance_improvement': relevance_improvement,
                'query_results': query_results
            }
            
    except Exception as e:
        print(f"   ❌ Phase 1 test failed: {e}")
        return None

if __name__ == "__main__":
    results = test_phase1_improvements()
    
    if results:
        with open('tests/phase1_improvement_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\n💾 Results saved to 'tests/phase1_improvement_results.json'")
        print("🎯 Phase 1 testing complete!")
    else:
        print("❌ Phase 1 testing failed!")