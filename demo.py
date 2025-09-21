#!/usr/bin/env python3
"""
AI Wellness Assistant Demo Script
Quick demonstration of core functionality
"""

import time
from datetime import datetime

def demo_stress_detection():
    """Demonstrate stress detection capabilities"""
    print("🧠 AI Wellness Assistant - Demo Mode")
    print("=" * 50)
    
    # Simulate stress detection results
    demo_results = [
        {"timestamp": "10:00:00", "stress": 0.2, "status": "Relaxed", "patterns": []},
        {"timestamp": "10:00:05", "stress": 0.4, "status": "Slight Tension", "patterns": ["concentration"]},
        {"timestamp": "10:00:10", "stress": 0.7, "status": "Moderate Stress", "patterns": ["tension", "anxiety"]},
        {"timestamp": "10:00:15", "stress": 0.9, "status": "High Stress - Action Needed", "patterns": ["tension", "anxiety", "frustration"]},
        {"timestamp": "10:00:20", "stress": 0.6, "status": "Improving", "patterns": ["tension"]},
        {"timestamp": "10:00:25", "stress": 0.3, "status": "Much Better", "patterns": []},
    ]
    
    print("📊 Simulated Real-time Stress Analysis:")
    print("-" * 50)
    
    for result in demo_results:
        stress_bar = "█" * int(result["stress"] * 20) + "░" * (20 - int(result["stress"] * 20))
        
        if result["stress"] < 0.4:
            color = "🔵"  # Blue for low stress
        elif result["stress"] < 0.7:
            color = "🟡"  # Yellow for moderate
        else:
            color = "🔴"  # Red for high stress
        
        print(f"{result['timestamp']} {color} [{stress_bar}] {result['stress']:.1%} - {result['status']}")
        
        # Show detected patterns
        if result["patterns"]:
            patterns_str = ", ".join(result["patterns"])
            print(f"    📝 Patterns: {patterns_str}")
        
        # Show FACS action units (simulated)
        if result["stress"] > 0.5:
            action_units = ["AU4 (Brow Lowerer)", "AU7 (Lid Tightener)", "AU23 (Lip Tightener)"]
            print(f"    🔍 FACS: {', '.join(action_units[:2])}")
        
        # Simulate recommendations for high stress
        if result["stress"] > 0.6:
            print("    💡 Recommendation: Try 4-7-8 breathing technique")
            print("    🧘 Inhale 4s → Hold 7s → Exhale 8s")
            if result["stress"] > 0.8:
                print("    ⚠️  High stress detected! Consider taking a break.")
        
        time.sleep(1.5)  # Simulate real-time delay
    
    print("\n🎯 Demo Complete!")
    print("📋 Key Features Demonstrated:")
    print("  ✓ Real-time stress quantification (0-100%)")
    print("  ✓ FACS action unit detection")
    print("  ✓ Pattern recognition (tension, anxiety, frustration)")
    print("  ✓ Personalized recommendations")
    print("  ✓ Progressive stress tracking")
    print("  ✓ Immediate intervention alerts")
    
    print("\n🚀 To run full application:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run application: python app.py")
    print("  3. Open browser: http://localhost:5000")
    
    print("\n🔗 Project Links:")
    print("  GitHub: https://github.com/Gzeu/ai-wellness-assistant")
    print("  Linear: View tasks and progress")
    print("  Notion: Complete documentation")

def demo_wellness_techniques():
    """Demonstrate wellness coaching features"""
    print("\n🧘 Wellness Coaching Demo")
    print("=" * 30)
    
    techniques = [
        {
            "name": "4-7-8 Breathing Technique",
            "description": "Inhale for 4, hold for 7, exhale for 8 seconds",
            "stress_levels": ["moderate", "high", "very_high"]
        },
        {
            "name": "5-4-3-2-1 Grounding",
            "description": "Use your senses to anchor in the present",
            "stress_levels": ["slight", "moderate", "high"]
        },
        {
            "name": "Progressive Muscle Relaxation",
            "description": "Systematically tense and release muscle groups",
            "stress_levels": ["moderate", "high", "very_high"]
        }
    ]
    
    for i, technique in enumerate(techniques, 1):
        print(f"\n{i}. 🎯 {technique['name']}")
        print(f"   Description: {technique['description']}")
        print(f"   Best for: {', '.join(technique['stress_levels'])} stress")
    
    print("\n✨ All techniques are evidence-based and scientifically validated!")

if __name__ == "__main__":
    demo_stress_detection()
    demo_wellness_techniques()