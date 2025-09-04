#!/usr/bin/env python3
"""
Test Original Validation Queries - Exact Match
Use the exact same 3 queries from the original validation to isolate query differences
"""

import json
import statistics
import time
from typing import List, Dict, Any

def test_original_validation_queries():
    """Test using the exact same 3 queries from original validation"""
    
    print("🚀 TESTING ORIGINAL VALIDATION QUERIES")
    print("=" * 50)
    
    # EXACT SAME queries from validate_all_implementations.py that achieved 91.7%
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
        }
    ]
    
    print(f"🔍 Testing with ORIGINAL {len(test_queries)} validation queries...")
    
    try:
        from medibotllamaindex import hybrid_retriever
        
        query_results = []
        
        for query_info in test_queries:
            query = query_info["query"]
            expected_keywords = query_info["expected_keywords"]
            category = query_info["category"]
            
            print(f"\\n   🔍 Testing: {query}")
            print(f"      Category: {category}")
            
            try:
                # Test retrieval performance using pure original query
                start_time = time.time()
                nodes = hybrid_retriever.retrieve(query)
                retrieval_time = time.time() - start_time
                
                relevance_scores = []
                high_relevance_count = 0
                
                for node in nodes:
                    content = node.node.text.lower()
                    
                    # Standard relevance calculation (same as original validation)
                    matches = [kw for kw in expected_keywords if kw.lower() in content]
                    relevance_score = len(matches) / len(expected_keywords)
                    relevance_scores.append(relevance_score)
                    
                    if relevance_score >= 0.5:
                        high_relevance_count += 1
                
                precision = high_relevance_count / len(nodes) if nodes else 0.0
                avg_relevance = statistics.mean(relevance_scores) if relevance_scores else 0.0
                
                query_results.append({
                    'query': query,
                    'category': category,
                    'precision': precision,
                    'avg_relevance': avg_relevance,
                    'retrieval_time': retrieval_time,
                    'nodes_count': len(nodes)
                })
                
                print(f"      📊 P={precision:.3f} ({precision*100:.1f}%), R={avg_relevance:.3f} ({avg_relevance*100:.1f}%)")
                print(f"      📄 Retrieved {len(nodes)} nodes in {retrieval_time:.3f}s")
                
            except Exception as e:
                print(f"      ❌ Query failed: {e}")
                continue
        
        if query_results:
            # Calculate overall metrics (same as original)
            avg_precision = statistics.mean([r['precision'] for r in query_results])
            avg_relevance = statistics.mean([r['avg_relevance'] for r in query_results])
            avg_time = statistics.mean([r['retrieval_time'] for r in query_results])
            
            print(f"\\n🏆 ORIGINAL VALIDATION QUERY RESULTS:")
            print(f"   📈 Precision: {avg_precision:.4f} ({avg_precision*100:.1f}%)")
            print(f"   📈 Relevance: {avg_relevance:.4f} ({avg_relevance*100:.1f}%)")
            print(f"   ⚡ Speed: {avg_time:.3f}s")
            
            # Compare with expected original results
            expected_precision = 91.7  # Original hybrid_llamaindex result
            expected_relevance = 76.7
            
            precision_diff = (avg_precision * 100) - expected_precision
            relevance_diff = (avg_relevance * 100) - expected_relevance
            
            print(f"\\n📊 VALIDATION vs ORIGINAL:")
            print(f"   🎯 Expected: {expected_precision:.1f}% precision, {expected_relevance:.1f}% relevance")
            print(f"   📊 Current:  {avg_precision*100:.1f}% precision ({precision_diff:+.1f}%), {avg_relevance*100:.1f}% relevance ({relevance_diff:+.1f}%)")
            
            # Root cause analysis
            if abs(precision_diff) <= 1.0:
                print(f"\\n   ✅ QUERY FACTOR RULED OUT: Same queries give same results")
                print(f"      → Issue is NOT different test queries")
                print(f"      → Root cause is in document processing or retrieval setup")
            elif abs(precision_diff) <= 3.0:
                print(f"\\n   ⚠️  MINOR QUERY IMPACT: Small difference might be query-related")
            else:
                print(f"\\n   📊 QUERY FACTOR CONFIRMED: Different queries explain some performance gap")
                remaining_gap = precision_diff
                print(f"      → Query difference accounts for {remaining_gap:.1f}% of performance gap")
                print(f"      → Additional {6.0 - abs(remaining_gap):.1f}% gap may be from document processing")
            
            return {
                'avg_precision': avg_precision,
                'avg_relevance': avg_relevance,
                'avg_time': avg_time,
                'precision_diff': precision_diff,
                'relevance_diff': relevance_diff,
                'query_results': query_results
            }
            
    except Exception as e:
        print(f"   ❌ Original validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = test_original_validation_queries()
    
    if results:
        with open('tests/original_validation_repro_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\n💾 Results saved to 'tests/original_validation_repro_results.json'")
        print("🎯 Original validation query testing complete!")
        print("🔍 Root cause isolation step 1 finished!")
    else:
        print("❌ Original validation reproduction failed!")