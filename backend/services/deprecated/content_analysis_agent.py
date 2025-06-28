"""
LLM-powered Content Analysis Agent for intelligent article processing.

This agent is responsible ONLY for:
- Analyzing article content quality and relevance
- Generating intelligent summaries using LLM
- Extracting key insights and topics
- Rating content value and difficulty
- Categorizing articles by type and domain

Agent Boundaries:
- Does NOT handle data storage or retrieval
- Does NOT manage user interactions
- Does NOT perform search operations
- Does NOT handle workflow orchestration
"""

import json
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
from .article_content_fetcher import ArticleContentFetcher

# For local LLM integration
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Transformers not available - using fallback content analysis")


@dataclass
class ContentAnalysis:
    """Structured analysis result from the content analysis agent."""
    
    # Core Analysis
    quality_score: float  # 0.0 to 1.0
    relevance_score: float  # 0.0 to 1.0
    difficulty_level: str  # "beginner", "intermediate", "advanced", "expert"
    estimated_read_time: int  # in minutes
    
    # Content Classification
    primary_category: str
    secondary_categories: List[str]
    content_type: str  # "tutorial", "opinion", "news", "research", "case_study", etc.
    
    # Extracted Intelligence
    key_insights: List[str]
    main_topics: List[str]
    technologies_mentioned: List[str]
    actionable_takeaways: List[str]
    
    # Enhanced Summary
    intelligent_summary: str
    key_quotes: List[str]
    
    # Metadata
    analysis_timestamp: str
    analysis_version: str = "1.0"
    confidence_score: float = 0.0


class ContentAnalysisAgent:
    """
    Intelligent Content Analysis Agent using LLM for deep content understanding.
    
    Agent Responsibilities:
    - Analyze article content for quality and relevance
    - Generate intelligent, contextual summaries
    - Extract key insights and actionable information
    - Classify content by type, difficulty, and domain
    - Rate content value for different user types
    
    Agent Boundaries:
    - Only processes content, doesn't store or retrieve data
    - Doesn't handle user preferences or personalization
    - Doesn't manage workflows or orchestration
    """
    
    def __init__(self, use_local_llm: bool = True, model_name: str = "microsoft/DialoGPT-medium"):
        self.use_local_llm = use_local_llm and HAS_TRANSFORMERS
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.generator = None
        
        # Analysis templates for different content types
        self.analysis_templates = {
            "technical": {
                "focus": ["implementation details", "code examples", "technical concepts", "tools and frameworks"],
                "quality_indicators": ["code quality", "explanation clarity", "practical examples", "completeness"]
            },
            "business": {
                "focus": ["business value", "strategic insights", "market analysis", "ROI considerations"],
                "quality_indicators": ["data backing", "real-world examples", "actionable advice", "credibility"]
            },
            "educational": {
                "focus": ["learning outcomes", "step-by-step guidance", "concept clarity", "practical exercises"],
                "quality_indicators": ["pedagogical structure", "examples quality", "progression logic", "accessibility"]
            }
        }
        
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the local LLM for content analysis."""
        if not self.use_local_llm:
            print("[CONTENT_AGENT] Using rule-based fallback analysis (no LLM)")
            return
        
        try:
            print(f"[CONTENT_AGENT] Initializing local LLM: {self.model_name}")
            
            # Use a lightweight model for content analysis
            # In production, you might use a more powerful model
            self.generator = pipeline(
                "text-generation",
                model="distilgpt2",  # Lightweight model
                device=0 if torch.cuda.is_available() else -1,
                max_length=512,
                truncation=True
            )
            
            print("[CONTENT_AGENT] ✓ LLM initialized successfully")
            
        except Exception as e:
            print(f"[CONTENT_AGENT] Failed to initialize LLM: {e}")
            print("[CONTENT_AGENT] Falling back to rule-based analysis")
            self.use_local_llm = False
    
    async def analyze_article_content(self, article: Dict, medium_cookies: Dict[str, str] = None) -> ContentAnalysis:
        """
        Perform comprehensive content analysis on an article.
        
        Args:
            article: Article dictionary with title and link
            medium_cookies: Medium session cookies for accessing member content
            
        Returns:
            ContentAnalysis object with detailed intelligence
        """
        try:
            print(f"[CONTENT_AGENT] Analyzing article: {article.get('title', 'Unknown')[:50]}...")
            
            title = article.get('title', '')
            link = article.get('link', '')
            
            # Step 1: Fetch the full article content from Medium
            print(f"[CONTENT_AGENT] Fetching full content from: {link}")
            fetcher = ArticleContentFetcher(session_cookies=medium_cookies or {})
            full_content = fetcher.fetch_article_content(link)
            
            if not full_content:
                print(f"[CONTENT_AGENT] Failed to fetch content, using fallback analysis")
                return self._create_fallback_analysis(article)
            
            print(f"[CONTENT_AGENT] ✓ Fetched {len(full_content)} characters of content")
            
            # Step 2: Analyze the full content
            if self.use_local_llm:
                analysis = await self._llm_based_analysis(title, full_content, link)
            else:
                analysis = await self._rule_based_analysis(title, full_content, link)
            
            print(f"[CONTENT_AGENT] ✓ Analysis complete - Quality: {analysis.quality_score:.2f}, Category: {analysis.primary_category}")
            return analysis
            
        except Exception as e:
            print(f"[CONTENT_AGENT] Error analyzing article: {e}")
            return self._create_fallback_analysis(article)
    
    async def _llm_based_analysis(self, title: str, content: str, link: str) -> ContentAnalysis:
        """Perform LLM-powered content analysis."""
        
        # Create analysis prompt
        analysis_prompt = f"""
        Analyze this article for quality, relevance, and key insights:
        
        Title: {title}
        Content: {content[:800]}...
        
        Provide analysis in the following areas:
        1. Quality (0.0-1.0): Technical accuracy, clarity, depth
        2. Category: Main topic category
        3. Type: tutorial/opinion/news/research/case_study
        4. Difficulty: beginner/intermediate/advanced/expert
        5. Key insights: 3 most important takeaways
        6. Technologies: Programming languages, tools, frameworks mentioned
        7. Summary: Enhanced 2-sentence summary
        """
        
        try:
            # Generate analysis using LLM
            response = self.generator(
                analysis_prompt,
                max_length=len(analysis_prompt) + 200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            # Parse LLM response and create structured analysis
            llm_output = response[0]['generated_text'][len(analysis_prompt):].strip()
            
            # For demo purposes, combine LLM insights with rule-based structure
            # In production, you'd have more sophisticated LLM parsing
            rule_analysis = await self._rule_based_analysis(title, content, link)
            
            # Enhance with LLM insights
            rule_analysis.intelligent_summary = self._extract_summary_from_llm(llm_output, content)
            rule_analysis.key_insights = self._extract_insights_from_llm(llm_output, title, content)
            rule_analysis.confidence_score = 0.8
            
            return rule_analysis
            
        except Exception as e:
            print(f"[CONTENT_AGENT] LLM analysis failed: {e}, falling back to rules")
            return await self._rule_based_analysis(title, content, link)
    
    async def _rule_based_analysis(self, title: str, content: str, link: str) -> ContentAnalysis:
        """Perform sophisticated rule-based content analysis."""
        
        # Analyze content type and domain
        content_type = self._classify_content_type(title, content)
        primary_category = self._classify_primary_category(title, content)
        difficulty_level = self._assess_difficulty_level(title, content)
        
        # Quality assessment
        quality_score = self._assess_content_quality(title, content, content_type)
        relevance_score = self._assess_relevance(title, content, primary_category)
        
        # Extract intelligence
        key_insights = self._extract_key_insights(title, content, content_type)
        main_topics = self._extract_main_topics(title, content)
        technologies = self._extract_technologies(title, content)
        takeaways = self._extract_actionable_takeaways(content, content_type)
        
        # Enhanced summary
        intelligent_summary = self._generate_intelligent_summary(title, content, key_insights)
        key_quotes = self._extract_key_quotes(content)
        
        # Estimate read time
        read_time = self._estimate_read_time(content)
        
        return ContentAnalysis(
            quality_score=quality_score,
            relevance_score=relevance_score,
            difficulty_level=difficulty_level,
            estimated_read_time=read_time,
            primary_category=primary_category,
            secondary_categories=self._get_secondary_categories(title, content),
            content_type=content_type,
            key_insights=key_insights,
            main_topics=main_topics,
            technologies_mentioned=technologies,
            actionable_takeaways=takeaways,
            intelligent_summary=intelligent_summary,
            key_quotes=key_quotes,
            analysis_timestamp=datetime.now().isoformat(),
            confidence_score=0.7
        )
    
    def _classify_content_type(self, title: str, content: str) -> str:
        """Classify the type of content."""
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Tutorial indicators
        if any(word in title_lower for word in ['guide', 'tutorial', 'how to', 'step by step', 'walkthrough']):
            return "tutorial"
        
        # Opinion/commentary indicators
        if any(word in title_lower for word in ['opinion', 'thoughts on', 'why i', 'my take', 'unpopular opinion']):
            return "opinion"
        
        # News/update indicators
        if any(word in title_lower for word in ['announced', 'released', 'new version', 'breaking', 'update']):
            return "news"
        
        # Research/analysis indicators
        if any(word in title_lower for word in ['analysis', 'research', 'study', 'comparison', 'benchmark']):
            return "research"
        
        # Case study indicators
        if any(word in title_lower for word in ['case study', 'experience', 'journey', 'lessons learned']):
            return "case_study"
        
        # Default based on content structure
        if len(content.split('.')) > 10 and any(word in content_lower for word in ['first', 'second', 'step', 'then']):
            return "tutorial"
        
        return "article"
    
    def _classify_primary_category(self, title: str, content: str) -> str:
        """Classify the primary topic category."""
        text = (title + " " + content).lower()
        
        # Technology categories
        if any(word in text for word in ['python', 'javascript', 'react', 'node', 'api', 'database', 'code']):
            return "Programming"
        
        if any(word in text for word in ['ai', 'machine learning', 'deep learning', 'neural', 'llm', 'gpt']):
            return "AI/ML"
        
        if any(word in text for word in ['data science', 'analytics', 'visualization', 'pandas', 'sql']):
            return "Data Science"
        
        if any(word in text for word in ['devops', 'docker', 'kubernetes', 'aws', 'cloud', 'deployment']):
            return "DevOps"
        
        # Business categories
        if any(word in text for word in ['startup', 'business', 'marketing', 'sales', 'strategy']):
            return "Business"
        
        if any(word in text for word in ['product', 'design', 'ux', 'ui', 'user experience']):
            return "Product Design"
        
        return "Technology"
    
    def _assess_difficulty_level(self, title: str, content: str) -> str:
        """Assess the difficulty level of the content."""
        text = (title + " " + content).lower()
        
        # Beginner indicators
        beginner_words = ['beginner', 'introduction', 'getting started', 'basics', 'fundamentals', 'simple']
        if any(word in text for word in beginner_words):
            return "beginner"
        
        # Advanced indicators
        advanced_words = ['advanced', 'expert', 'optimization', 'performance', 'architecture', 'scalability']
        if any(word in text for word in advanced_words):
            return "advanced"
        
        # Expert indicators
        expert_words = ['internals', 'deep dive', 'research', 'thesis', 'algorithm', 'complexity']
        if any(word in text for word in expert_words):
            return "expert"
        
        return "intermediate"
    
    def _assess_content_quality(self, title: str, content: str, content_type: str) -> float:
        """Assess content quality on a 0-1 scale."""
        score = 0.5  # Base score
        
        # Length indicates depth
        if len(content) > 500:
            score += 0.1
        if len(content) > 1000:
            score += 0.1
        
        # Structure indicators
        if content.count('.') > 5:  # Multiple sentences
            score += 0.1
        
        # Technical content indicators
        if content_type in ["tutorial", "research"]:
            if any(word in content.lower() for word in ['example', 'code', 'implementation']):
                score += 0.1
        
        # Clarity indicators
        if not any(word in title.lower() for word in ['clickbait', 'you won\'t believe', 'shocking']):
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_relevance(self, title: str, content: str, category: str) -> float:
        """Assess content relevance."""
        # For now, assume all categorized content is relevant
        # In a real system, this would be based on user preferences
        return 0.8 if category != "Technology" else 0.9
    
    def _extract_key_insights(self, title: str, content: str, content_type: str) -> List[str]:
        """Extract key insights from the content."""
        insights = []
        
        # Look for insight patterns in content
        sentences = content.split('.')
        for sentence in sentences[:5]:  # Check first 5 sentences
            sentence = sentence.strip()
            if len(sentence) > 50 and any(word in sentence.lower() for word in ['key', 'important', 'main', 'primary']):
                insights.append(sentence[:100] + "..." if len(sentence) > 100 else sentence)
        
        # Add domain-specific insights
        if 'ai' in content.lower() or 'machine learning' in content.lower():
            insights.append("Explores AI/ML applications and implications")
        
        if 'python' in content.lower():
            insights.append("Covers Python programming concepts and best practices")
        
        return insights[:3]  # Return top 3 insights
    
    def _extract_main_topics(self, title: str, content: str) -> List[str]:
        """Extract main topics from the content."""
        topics = []
        text = (title + " " + content).lower()
        
        # Technology topics
        tech_topics = {
            'python': 'Python', 'javascript': 'JavaScript', 'react': 'React', 
            'ai': 'Artificial Intelligence', 'machine learning': 'Machine Learning',
            'data science': 'Data Science', 'api': 'APIs', 'database': 'Databases'
        }
        
        for keyword, topic in tech_topics.items():
            if keyword in text:
                topics.append(topic)
        
        return topics[:5]
    
    def _extract_technologies(self, title: str, content: str) -> List[str]:
        """Extract mentioned technologies."""
        technologies = []
        text = (title + " " + content).lower()
        
        tech_list = [
            'python', 'javascript', 'typescript', 'react', 'vue', 'angular',
            'node.js', 'django', 'flask', 'fastapi', 'docker', 'kubernetes',
            'aws', 'azure', 'gcp', 'postgresql', 'mongodb', 'redis'
        ]
        
        for tech in tech_list:
            if tech in text:
                technologies.append(tech.title())
        
        return technologies[:5]
    
    def _extract_actionable_takeaways(self, content: str, content_type: str) -> List[str]:
        """Extract actionable takeaways."""
        takeaways = []
        
        if content_type == "tutorial":
            takeaways.append("Follow the step-by-step implementation guide")
            takeaways.append("Practice with the provided examples")
        
        elif content_type == "opinion":
            takeaways.append("Consider the alternative perspective presented")
            takeaways.append("Evaluate how this applies to your context")
        
        elif content_type == "research":
            takeaways.append("Review the methodology and findings")
            takeaways.append("Consider implications for your work")
        
        return takeaways
    
    def _generate_intelligent_summary(self, title: str, content: str, insights: List[str]) -> str:
        """Generate an enhanced, intelligent summary."""
        # Create a more intelligent summary than the basic one
        first_sentence = content.split('.')[0] if content else title
        
        if insights:
            return f"{first_sentence}. This article provides valuable insights including {insights[0] if insights else 'key concepts'} and practical guidance for implementation."
        
        return f"{first_sentence}. This article offers detailed coverage of the topic with practical examples and actionable advice."
    
    def _extract_key_quotes(self, content: str) -> List[str]:
        """Extract potentially quotable sentences."""
        quotes = []
        sentences = content.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Look for impactful statements
            if (len(sentence) > 30 and len(sentence) < 150 and 
                any(word in sentence.lower() for word in ['key', 'important', 'crucial', 'essential', 'vital'])):
                quotes.append(sentence)
        
        return quotes[:2]
    
    def _estimate_read_time(self, content: str) -> int:
        """Estimate reading time in minutes."""
        words = len(content.split())
        # Average reading speed: 200-250 words per minute
        return max(1, round(words / 225))
    
    def _get_secondary_categories(self, title: str, content: str) -> List[str]:
        """Get secondary categories for the content."""
        categories = []
        text = (title + " " + content).lower()
        
        category_keywords = {
            'Best Practices': ['best practice', 'tips', 'guidelines'],
            'Performance': ['performance', 'optimization', 'speed'],
            'Security': ['security', 'authentication', 'vulnerability'],
            'Architecture': ['architecture', 'design pattern', 'structure'],
            'Tools': ['tool', 'framework', 'library']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text for keyword in keywords):
                categories.append(category)
        
        return categories[:3]
    
    def _extract_summary_from_llm(self, llm_output: str, original_content: str) -> str:
        """Extract enhanced summary from LLM output."""
        # Simple extraction - in production you'd have better parsing
        if "summary:" in llm_output.lower():
            summary_part = llm_output.lower().split("summary:")[1].split("\n")[0]
            return summary_part.strip().capitalize()
        
        return self._generate_intelligent_summary("", original_content, [])
    
    def _extract_insights_from_llm(self, llm_output: str, title: str, content: str) -> List[str]:
        """Extract insights from LLM output."""
        # Simple extraction - in production you'd have better parsing
        if "insights:" in llm_output.lower():
            insights_part = llm_output.lower().split("insights:")[1].split("\n")[0]
            return [insights_part.strip().capitalize()]
        
        return self._extract_key_insights(title, content, "article")
    
    def _create_fallback_analysis(self, article: Dict) -> ContentAnalysis:
        """Create a basic analysis when all else fails."""
        return ContentAnalysis(
            quality_score=0.5,
            relevance_score=0.5,
            difficulty_level="intermediate",
            estimated_read_time=3,
            primary_category="General",
            secondary_categories=[],
            content_type="article",
            key_insights=["Content analysis not available"],
            main_topics=[],
            technologies_mentioned=[],
            actionable_takeaways=[],
            intelligent_summary=article.get('title', 'Article content') + " - Analysis unavailable.",
            key_quotes=[],
            analysis_timestamp=datetime.now().isoformat(),
            confidence_score=0.3
        )


# Global content analysis agent instance
content_analysis_agent = ContentAnalysisAgent(use_local_llm=False)  # Start with rule-based for stability
