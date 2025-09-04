#!/usr/bin/env python3
"""
Test Pure Baseline - Zero Modifications
Verify we can reproduce original 91.7% precision
"""

import json
import statistics
import time
from typing import List, Dict, Any

def test_pure_baseline():
    """Test pure baseline with zero modifications to validate original performance"""
    
    print("🚀 TESTING PURE BASELINE - ZERO MODIFICATIONS")
    print("=" * 55)
    
    # Same test queries used in final validation
    test_queries = [
        {
            "query": "How do I create an appointment in MediRecords?",
            "expected_keywords": ["appointment", "create", "schedule", "booking"]
        },
        {
            "query": "Steps to process Medicare billing claims?", 
            "expected_keywords": ["medicare", "billing", "claim", "payment"]
        },
        {
            "query": "Where is the patient registration located?",
            "expected_keywords": ["patient", "register", "location", "find"]
        },
        {
            "query": "EMR system not working properly",
            "expected_keywords": ["system", "error", "problem", "fix"]
        },
        {
            "query": "What are the clinical features in MediRecord?",
            "expected_keywords": ["clinical", "features", "medical", "capabilities"]
        },
        {
            "query": "Setup SMS reminders for appointments",
            "expected_keywords": ["sms", "reminder", "appointment", "setup"]
        },
        {
            "query": "Generate billing reports for patients", 
            "expected_keywords": ["report", "billing", "generate", "patient"]
        }
    ]
    
    print("🔍 Testing Pure Baseline Implementation...")
    
    try:
        from medibotllamaindex import hybrid_retriever
        
        query_results = []
        
        for query_info in test_queries:
            query = query_info["query"]
            expected_keywords = query_info["expected_keywords"]
            
            print(f"\\n   🔍 Testing: {query}")
            
            try:
                # Test retrieval performance using pure original query
                start_time = time.time()
                nodes = hybrid_retriever.retrieve(query)
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
            avg_time = statistics.mean([r['retrieval_time'] for r in query_results])
            
            print(f"\\n🏆 PURE BASELINE RESULTS:")
            print(f"   📈 Precision: {avg_precision:.3f} ({avg_precision*100:.1f}%)")
            print(f"   📈 Relevance: {avg_relevance:.3f} ({avg_relevance*100:.1f}%)")
            print(f"   📈 Content Quality: {avg_quality:.3f} ({avg_quality*100:.1f}%)")
            print(f"   ⚡ Speed: {avg_time:.3f}s")
            
            # Compare with expected baseline
            expected_precision = 91.7  # Original validation result
            expected_relevance = 76.7
            
            precision_diff = (avg_precision * 100) - expected_precision
            relevance_diff = (avg_relevance * 100) - expected_relevance
            
            print(f"\\n📊 BASELINE VALIDATION:")
            print(f"   🎯 Expected Precision: {expected_precision:.1f}%")
            print(f"   📊 Actual Precision: {avg_precision*100:.1f}% ({precision_diff:+.1f}%)")
            print(f"   🎯 Expected Relevance: {expected_relevance:.1f}%")
            print(f"   📊 Actual Relevance: {avg_relevance*100:.1f}% ({relevance_diff:+.1f}%)")
            
            # Validation assessment
            if abs(precision_diff) <= 2.0:
                print(f"\\n   ✅ VALIDATION SUCCESSFUL: Baseline performance reproduced!")
            elif abs(precision_diff) <= 5.0:
                print(f"\\n   ⚠️  CLOSE: Minor variance from expected baseline")
            else:
                print(f"\\n   ❌ VALIDATION FAILED: Significant deviation from baseline")
            
            return {
                'avg_precision': avg_precision,
                'avg_relevance': avg_relevance,
                'avg_quality': avg_quality,
                'avg_time': avg_time,
                'precision_diff': precision_diff,
                'relevance_diff': relevance_diff,
                'query_results': query_results
            }
            
    except Exception as e:
        print(f"   ❌ Pure baseline test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = test_pure_baseline()
    
    if results:
        with open('tests/pure_baseline_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\n💾 Results saved to 'tests/pure_baseline_results.json'")
        print("🎯 Pure baseline testing complete!")
    else:
        print("❌ Pure baseline testing failed!")