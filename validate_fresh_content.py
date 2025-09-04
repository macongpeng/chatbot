#!/usr/bin/env python3
"""
Quick validation test for Phase 2A+2B with fresh content
Verifies that precision improvements are maintained after knowledge base refresh
"""

import time
import json
from medibotllamaindex import getQueryResult

def run_quick_precision_test():
    """Run quick precision test with key queries"""
    
    print("🚀 VALIDATING PHASE 2A+2B WITH FRESH CONTENT")
    print("=" * 60)
    
    # Test queries from comprehensive baseline (key representative queries)
    test_queries = [
        {
            "query": "How do I create an appointment in MediRecords?",
            "difficulty": "easy",
            "expected_topics": ["appointment", "create", "schedule", "booking"]
        },
        {
            "query": "Where is the patient registration located?",
            "difficulty": "medium", 
            "expected_topics": ["patient", "registration", "add patient", "demographics"]
        },
        {
            "query": "How to generate Medicare reports?",
            "difficulty": "medium",
            "expected_topics": ["medicare", "reports", "generate", "billing"]
        },
        {
            "query": "EMR system not working properly",
            "difficulty": "hard",
            "expected_topics": ["system", "troubleshoot", "error", "technical"]
        }
    ]
    
    results = []
    total_score = 0
    
    print(f"\n🔍 Testing {len(test_queries)} key queries...\n")
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        difficulty = test_case["difficulty"]
        expected_topics = test_case["expected_topics"]
        
        print(f"[{i}] {difficulty.upper()}: {query}")
        
        start_time = time.time()
        try:
            result = getQueryResult(query)
            response_time = time.time() - start_time
            
            # Simple relevance scoring based on topic presence
            score = 0
            result_lower = result.lower()
            
            # Check for expected topics
            topic_matches = sum(1 for topic in expected_topics if topic in result_lower)
            if topic_matches >= 2:  # At least 2 expected topics
                score = 100
            elif topic_matches == 1:
                score = 75
            elif "don't know" not in result_lower and len(result) > 100:
                score = 50
            else:
                score = 25
                
            print(f"    ✓ Response time: {response_time:.3f}s")
            print(f"    ✓ Topic matches: {topic_matches}/{len(expected_topics)}")
            print(f"    ✓ Score: {score}/100")
            print(f"    → {result[:100]}...")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            score = 0
            response_time = 0
            result = f"Error: {e}"
        
        results.append({
            "query": query,
            "difficulty": difficulty,
            "score": score,
            "response_time": response_time,
            "result_preview": result[:200] + "..." if len(result) > 200 else result
        })
        
        total_score += score
        print()
    
    # Calculate overall metrics
    precision = total_score / (len(test_queries) * 100) * 100
    avg_response_time = sum(r["response_time"] for r in results) / len(results)
    
    print("📊 FRESH CONTENT VALIDATION RESULTS:")
    print("=" * 60)
    print(f"Overall Precision: {precision:.1f}%")
    print(f"Average Response Time: {avg_response_time:.3f}s")
    
    # Compare with expected Phase 2A+2B performance
    expected_precision = 85.7
    precision_diff = precision - expected_precision
    
    print(f"\n📈 COMPARISON WITH PHASE 2A+2B BASELINE:")
    print(f"Expected Precision: {expected_precision}%")
    print(f"Fresh Content Precision: {precision:.1f}%")
    print(f"Difference: {precision_diff:+.1f}%")
    
    if precision >= expected_precision - 2:  # Allow 2% tolerance
        print("✅ VALIDATION PASSED: Fresh content maintains Phase 2A+2B performance!")
    else:
        print("⚠️  VALIDATION WARNING: Performance below expected baseline")
    
    # Save results
    validation_results = {
        "test_type": "fresh_content_validation",
        "timestamp": time.time(),
        "total_queries": len(test_queries),
        "overall_precision": precision,
        "average_response_time": avg_response_time,
        "expected_precision": expected_precision,
        "precision_difference": precision_diff,
        "status": "PASSED" if precision >= expected_precision - 2 else "WARNING",
        "detailed_results": results
    }
    
    with open('tests/fresh_content_validation.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n💾 Results saved to tests/fresh_content_validation.json")
    return validation_results

if __name__ == "__main__":
    validation_results = run_quick_precision_test()