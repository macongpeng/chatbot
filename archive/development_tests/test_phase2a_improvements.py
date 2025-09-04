#!/usr/bin/env python3
"""
Test Phase 2A - Automated Content Intelligence
Measure performance improvements from automated metadata extraction
"""

import json
import statistics
import time
from typing import List, Dict, Any

def test_phase2a_improvements():
    """Test Phase 2A automated content intelligence improvements"""
    
    print("🚀 TESTING PHASE 2A - AUTOMATED CONTENT INTELLIGENCE")
    print("=" * 65)
    
    # Enhanced test queries to showcase category matching
    test_queries = [
        {
            "query": "How do I create an appointment in MediRecords?",
            "expected_keywords": ["appointment", "create", "schedule", "booking", "calendar"],
            "expected_category": "appointments",
            "expected_type": "procedure"
        },
        {
            "query": "What are the steps to process Medicare billing?", 
            "expected_keywords": ["medicare", "billing", "claim", "payment", "submit"],
            "expected_category": "billing", 
            "expected_type": "procedure"
        },
        {
            "query": "How do I add a new patient to the system?",
            "expected_keywords": ["patient", "add", "new", "register", "create"],
            "expected_category": "patients",
            "expected_type": "procedure"
        },
        {
            "query": "How do I set up SMS appointment reminders?",
            "expected_keywords": ["sms", "reminder", "appointment", "notification", "setup"],
            "expected_category": "reminders",
            "expected_type": "procedure"
        },
        {
            "query": "How to generate patient billing reports?",
            "expected_keywords": ["report", "billing", "generate", "export", "patient"],
            "expected_category": "reports",
            "expected_type": "procedure"
        },
        {
            "query": "What clinical features are available in MediRecords?",
            "expected_keywords": ["clinical", "features", "medical", "consultation", "record"],
            "expected_category": "clinical", 
            "expected_type": "overview"
        }
    ]
    
    print("🔍 Testing Phase 2A Enhanced Champion Implementation...")
    
    try:
        from medibotllamaindex import hybrid_retriever, preprocess_medical_query
        
        query_results = []
        category_accuracy = []
        content_type_accuracy = []
        
        for query_info in test_queries:
            query = query_info["query"]
            expected_keywords = query_info["expected_keywords"]
            expected_category = query_info["expected_category"]
            expected_type = query_info["expected_type"]
            
            print(f"\\n   🔍 Testing: {query}")
            print(f"      Expected: {expected_category}/{expected_type}")
            
            try:
                start_time = time.time()
                processed_query = preprocess_medical_query(query)
                nodes = hybrid_retriever.retrieve(processed_query)
                retrieval_time = time.time() - start_time
                
                relevance_scores = []
                high_relevance_count = 0
                quality_scores = []
                categories_found = []
                content_types_found = []
                
                for node in nodes:
                    content = node.node.text.lower()
                    metadata = node.node.metadata
                    
                    # Standard relevance calculation
                    matches = [kw for kw in expected_keywords if kw.lower() in content]
                    relevance_score = len(matches) / len(expected_keywords)
                    relevance_scores.append(relevance_score)
                    
                    # Quality and metadata tracking
                    quality_score = metadata.get('quality_score', 0.5)
                    quality_scores.append(quality_score)
                    
                    category = metadata.get('category', 'general')
                    content_type = metadata.get('content_type', 'reference')
                    categories_found.append(category)
                    content_types_found.append(content_type)
                    
                    if relevance_score >= 0.5:
                        high_relevance_count += 1
                
                precision = high_relevance_count / len(nodes) if nodes else 0.0
                avg_relevance = statistics.mean(relevance_scores) if relevance_scores else 0.0
                avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0
                
                # Phase 2A specific metrics
                category_match = categories_found.count(expected_category) / len(categories_found) if categories_found else 0.0
                content_type_match = content_types_found.count(expected_type) / len(content_types_found) if content_types_found else 0.0
                
                category_accuracy.append(category_match)
                content_type_accuracy.append(content_type_match)
                
                query_results.append({
                    'query': query,
                    'precision': precision,
                    'avg_relevance': avg_relevance,
                    'avg_quality': avg_quality,
                    'category_accuracy': category_match,
                    'content_type_accuracy': content_type_match,
                    'retrieval_time': retrieval_time,
                    'nodes_count': len(nodes)
                })
                
                print(f"      📊 P={precision:.3f}, R={avg_relevance:.3f}, Q={avg_quality:.3f}")
                print(f"      🏷️  Cat:{category_match:.3f} ({categories_found[0] if categories_found else 'none'})")
                print(f"      📝 Type:{content_type_match:.3f} ({content_types_found[0] if content_types_found else 'none'})")
                
            except Exception as e:
                print(f"      ❌ Query failed: {e}")
                continue
        
        if query_results:
            avg_precision = statistics.mean([r['precision'] for r in query_results])
            avg_relevance = statistics.mean([r['avg_relevance'] for r in query_results])
            avg_quality = statistics.mean([r['avg_quality'] for r in query_results])
            avg_category_accuracy = statistics.mean(category_accuracy)
            avg_content_type_accuracy = statistics.mean(content_type_accuracy)
            avg_time = statistics.mean([r['retrieval_time'] for r in query_results])
            
            print(f"\\n🏆 PHASE 2A ENHANCED RESULTS:")
            print(f"   📈 Precision: {avg_precision:.3f} ({avg_precision*100:.1f}%)")
            print(f"   📈 Relevance: {avg_relevance:.3f} ({avg_relevance*100:.1f}%)")
            print(f"   📈 Content Quality: {avg_quality:.3f} ({avg_quality*100:.1f}%)")
            print(f"   🏷️  Category Accuracy: {avg_category_accuracy:.3f} ({avg_category_accuracy*100:.1f}%)")
            print(f"   📝 Content Type Accuracy: {avg_content_type_accuracy:.3f} ({avg_content_type_accuracy*100:.1f}%)")
            print(f"   ⚡ Speed: {avg_time:.3f}s")
            
            # Compare with baseline
            baseline_precision = 91.7
            baseline_relevance = 76.7
            
            precision_improvement = (avg_precision * 100) - baseline_precision
            relevance_improvement = (avg_relevance * 100) - baseline_relevance
            
            print(f"\\n📊 PHASE 2A IMPROVEMENT ANALYSIS:")
            print(f"   🔥 Precision Change: {precision_improvement:+.1f}% ({baseline_precision:.1f}% → {avg_precision*100:.1f}%)")
            print(f"   🔥 Relevance Change: {relevance_improvement:+.1f}% ({baseline_relevance:.1f}% → {avg_relevance*100:.1f}%)")
            print(f"   🎯 NEW METRICS:")
            print(f"      • Category Matching: {avg_category_accuracy*100:.1f}% accuracy")
            print(f"      • Content Type Detection: {avg_content_type_accuracy*100:.1f}% accuracy")
            
            if precision_improvement >= 2.0:
                print(f"   🎉 EXCELLENT: Exceeded Phase 2A target (+1-2% precision)!")
            elif precision_improvement >= 1.0:
                print(f"   ✅ GOOD: Solid Phase 2A improvement achieved!")
            elif precision_improvement >= 0.0:
                print(f"   👍 POSITIVE: Phase 2A showing improvement!")
            else:
                print(f"   🔧 ANALYSIS: Phase 2A foundation working, may need tuning")
            
            # Phase 2A specific achievements
            print(f"\\n🚀 PHASE 2A ACHIEVEMENTS:")
            if avg_category_accuracy >= 0.6:
                print(f"   ✅ EXCELLENT category detection and matching")
            if avg_content_type_accuracy >= 0.7:
                print(f"   ✅ STRONG content type classification")
            if avg_quality >= 0.8:
                print(f"   ✅ HIGH quality content prioritization")
            
            return {
                'avg_precision': avg_precision,
                'avg_relevance': avg_relevance,
                'avg_quality': avg_quality,
                'avg_category_accuracy': avg_category_accuracy,
                'avg_content_type_accuracy': avg_content_type_accuracy,
                'avg_time': avg_time,
                'precision_improvement': precision_improvement,
                'relevance_improvement': relevance_improvement,
                'query_results': query_results
            }
            
    except Exception as e:
        print(f"   ❌ Phase 2A test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = test_phase2a_improvements()
    
    if results:
        with open('tests/phase2a_improvement_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\n💾 Results saved to 'tests/phase2a_improvement_results.json'")
        print("🎯 Phase 2A testing complete!")
        print("🚀 Ready for Phase 2B - Enhanced Query Processing!")
    else:
        print("❌ Phase 2A testing failed!")