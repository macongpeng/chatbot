#!/usr/bin/env python3
"""
MediRecords Knowledge Base Data Quality Analysis
Comprehensive analysis of source data quality issues
"""

import json
import os
import re
from typing import Dict, List, Tuple
from collections import defaultdict
import statistics

def analyze_data_quality() -> Dict:
    """Comprehensive data quality analysis"""
    
    knowledge_path = "htmlpages/knowledge/official/"
    
    if not os.path.exists(knowledge_path):
        return {"error": "Knowledge base path not found"}
    
    analysis = {
        "total_files": 0,
        "file_sizes": [],
        "quality_issues": defaultdict(list),
        "content_analysis": {
            "very_small_files": [],  # <300 bytes
            "small_files": [],       # 300-500 bytes  
            "medium_files": [],      # 500-2000 bytes
            "large_files": [],       # >2000 bytes
        },
        "title_analysis": {
            "titles_with_issues": [],
            "duplicate_titles": defaultdict(list)
        },
        "content_issues": {
            "incomplete_content": [],
            "generic_placeholders": [],
            "very_short_content": [],
            "malformed_json": []
        }
    }
    
    print("🔍 Starting comprehensive data quality analysis...")
    print("=" * 60)
    
    files = [f for f in os.listdir(knowledge_path) if os.path.isfile(os.path.join(knowledge_path, f))]
    analysis["total_files"] = len(files)
    
    for filename in files:
        filepath = os.path.join(knowledge_path, filename)
        file_size = os.path.getsize(filepath)
        analysis["file_sizes"].append(file_size)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            title = data.get('title', '').strip()
            body = data.get('body', '').strip()
            
            # Categorize by file size
            if file_size < 300:
                analysis["content_analysis"]["very_small_files"].append({
                    "filename": filename,
                    "size": file_size,
                    "title": title,
                    "body_length": len(body)
                })
            elif file_size < 500:
                analysis["content_analysis"]["small_files"].append({
                    "filename": filename, 
                    "size": file_size,
                    "title": title,
                    "body_length": len(body)
                })
            elif file_size < 2000:
                analysis["content_analysis"]["medium_files"].append({
                    "filename": filename,
                    "size": file_size,
                    "title": title,
                    "body_length": len(body)
                })
            else:
                analysis["content_analysis"]["large_files"].append({
                    "filename": filename,
                    "size": file_size, 
                    "title": title,
                    "body_length": len(body)
                })
            
            # Title analysis
            if title:
                analysis["title_analysis"]["duplicate_titles"][title].append(filename)
            else:
                analysis["title_analysis"]["titles_with_issues"].append({
                    "filename": filename,
                    "issue": "Missing title"
                })
            
            # Content quality analysis
            if len(body) < 100:
                analysis["content_issues"]["very_short_content"].append({
                    "filename": filename,
                    "title": title,
                    "body_length": len(body),
                    "content": body[:200] + "..." if len(body) > 200 else body
                })
            
            # Check for incomplete content patterns
            incomplete_patterns = [
                "for more information.*see the following articles",
                "click here for more details",
                "please contact support",
                "more information.*found.*here"
            ]
            
            for pattern in incomplete_patterns:
                if re.search(pattern, body, re.IGNORECASE):
                    analysis["content_issues"]["incomplete_content"].append({
                        "filename": filename,
                        "title": title, 
                        "pattern_matched": pattern,
                        "body_preview": body[:200] + "..." if len(body) > 200 else body
                    })
            
            # Check for generic placeholders
            placeholder_patterns = [
                r"\\n\\s*\\n",  # Multiple empty lines
                "lorem ipsum",
                "placeholder",
                "todo",
                "tbd",
                "coming soon"
            ]
            
            for pattern in placeholder_patterns:
                if re.search(pattern, body, re.IGNORECASE):
                    analysis["content_issues"]["generic_placeholders"].append({
                        "filename": filename,
                        "title": title,
                        "placeholder_type": pattern
                    })
                    
        except json.JSONDecodeError as e:
            analysis["content_issues"]["malformed_json"].append({
                "filename": filename,
                "error": str(e)
            })
        except Exception as e:
            analysis["quality_issues"]["file_read_errors"].append({
                "filename": filename,
                "error": str(e)
            })
    
    # Calculate statistics
    if analysis["file_sizes"]:
        analysis["size_statistics"] = {
            "min_size": min(analysis["file_sizes"]),
            "max_size": max(analysis["file_sizes"]),
            "average_size": statistics.mean(analysis["file_sizes"]),
            "median_size": statistics.median(analysis["file_sizes"]),
        }
    
    # Find duplicate titles
    duplicates = {title: files for title, files in analysis["title_analysis"]["duplicate_titles"].items() 
                  if len(files) > 1}
    analysis["title_analysis"]["duplicate_titles"] = duplicates
    
    return analysis

def print_quality_summary(analysis: Dict):
    """Print formatted quality analysis summary"""
    
    print(f"\\n📊 DATA QUALITY ANALYSIS SUMMARY")
    print("=" * 50)
    
    print(f"\\n📁 OVERALL STATISTICS:")
    print(f"   Total Files: {analysis['total_files']}")
    
    if "size_statistics" in analysis:
        stats = analysis["size_statistics"]
        print(f"   File Size Range: {stats['min_size']}-{stats['max_size']} bytes")
        print(f"   Average Size: {stats['average_size']:.0f} bytes")
        print(f"   Median Size: {stats['median_size']:.0f} bytes")
    
    print(f"\\n🗂️  FILE SIZE DISTRIBUTION:")
    content = analysis["content_analysis"]
    print(f"   Very Small (<300B): {len(content['very_small_files'])} files ({len(content['very_small_files'])/analysis['total_files']*100:.1f}%)")
    print(f"   Small (300-500B): {len(content['small_files'])} files ({len(content['small_files'])/analysis['total_files']*100:.1f}%)")
    print(f"   Medium (500-2000B): {len(content['medium_files'])} files ({len(content['medium_files'])/analysis['total_files']*100:.1f}%)")
    print(f"   Large (>2000B): {len(content['large_files'])} files ({len(content['large_files'])/analysis['total_files']*100:.1f}%)")
    
    print(f"\\n🚨 QUALITY ISSUES DETECTED:")
    issues = analysis["content_issues"]
    print(f"   Very Short Content (<100 chars): {len(issues['very_short_content'])} files")
    print(f"   Incomplete Content: {len(issues['incomplete_content'])} files")
    print(f"   Generic Placeholders: {len(issues['generic_placeholders'])} files")
    print(f"   Malformed JSON: {len(issues['malformed_json'])} files")
    
    if analysis["title_analysis"]["duplicate_titles"]:
        print(f"   Duplicate Titles: {len(analysis['title_analysis']['duplicate_titles'])} sets")
    
    # Show worst offenders
    if issues["very_short_content"]:
        print(f"\\n⚠️  VERY SHORT CONTENT EXAMPLES:")
        for item in issues["very_short_content"][:3]:
            print(f"   • {item['title']} ({item['body_length']} chars)")
            print(f"     Content: {item['content'][:100]}...")
    
    if issues["incomplete_content"]:
        print(f"\\n❌ INCOMPLETE CONTENT EXAMPLES:")
        for item in issues["incomplete_content"][:3]:
            print(f"   • {item['title']}")
            print(f"     Issue: {item['pattern_matched']}")
            print(f"     Preview: {item['body_preview'][:100]}...")

def calculate_data_quality_score(analysis: Dict) -> float:
    """Calculate overall data quality score (0-100)"""
    
    total_files = analysis["total_files"]
    if total_files == 0:
        return 0.0
    
    # Penalty factors
    penalties = 0
    
    # Size penalties
    very_small_penalty = len(analysis["content_analysis"]["very_small_files"]) * 3
    small_penalty = len(analysis["content_analysis"]["small_files"]) * 1
    
    # Content penalties  
    short_content_penalty = len(analysis["content_issues"]["very_short_content"]) * 2
    incomplete_penalty = len(analysis["content_issues"]["incomplete_content"]) * 4
    placeholder_penalty = len(analysis["content_issues"]["generic_placeholders"]) * 2
    malformed_penalty = len(analysis["content_issues"]["malformed_json"]) * 5
    
    total_penalties = (very_small_penalty + small_penalty + short_content_penalty + 
                      incomplete_penalty + placeholder_penalty + malformed_penalty)
    
    # Calculate score (max penalty normalized to total files)
    max_possible_penalty = total_files * 5  # Worst case: all files have max penalty
    penalty_ratio = min(total_penalties / max_possible_penalty, 1.0) if max_possible_penalty > 0 else 0
    
    quality_score = max(0, 100 - (penalty_ratio * 100))
    
    return quality_score

if __name__ == "__main__":
    analysis_results = analyze_data_quality()
    
    if "error" in analysis_results:
        print(f"❌ Error: {analysis_results['error']}")
    else:
        print_quality_summary(analysis_results)
        
        quality_score = calculate_data_quality_score(analysis_results)
        print(f"\\n🎯 OVERALL DATA QUALITY SCORE: {quality_score:.1f}/100")
        
        if quality_score >= 80:
            print("✅ EXCELLENT data quality")
        elif quality_score >= 60:
            print("⚠️  GOOD data quality with room for improvement")
        elif quality_score >= 40:
            print("🚨 POOR data quality - significant issues detected")
        else:
            print("💥 CRITICAL data quality issues - major remediation needed")
        
        # Save detailed results
        with open('tests/data_quality_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"\\n💾 Detailed analysis saved to 'tests/data_quality_analysis.json'")