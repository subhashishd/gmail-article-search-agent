"""Streamlit frontend application for Gmail Article Search Agent."""

import streamlit as st
import pandas as pd
import requests
import time
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BACKEND_URL = os.getenv("BACKEND_SERVICE_URL", "http://localhost:8000")

# Configure Streamlit page
st.set_page_config(
    page_title="Gmail Article Search",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

def call_backend_api(endpoint: str, method: str = "GET", data: dict = None):
    """
    Make API calls to the backend service with cache-busting headers.
    
    Args:
        endpoint: API endpoint
        method: HTTP method
        data: Request data for POST requests
        
    Returns:
        API response or None if error
    """
    try:
        url = f"{BACKEND_URL}{endpoint}"
        
        # Add cache-busting headers and timestamp parameter to prevent stale responses
        cache_buster = str(int(time.time()))
        headers = {
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'X-Cache-Bust': cache_buster
        }
        
        # Add timestamp parameter to URL to force cache refresh
        separator = '&' if '?' in url else '?'
        url = f"{url}{separator}_t={cache_buster}"
        
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=30)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            return None
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Could not connect to backend service at {BACKEND_URL}")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out")
        return None
    except Exception as e:
        st.error(f"❌ Error calling API: {str(e)}")
        return None

def display_articles_as_cards(articles: list):
    """
    Display articles in a clean, readable card format with paragraph-style layout.
    
    Args:
        articles: List of article dictionaries
    """
    if not articles:
        st.info("No articles found.")
        return
    
    st.markdown("### 📚 Search Results")
    
    # Track seen articles to handle duplicates
    seen_articles = set()
    unique_articles = []
    
    for article in articles:
        # Create a unique identifier for each article
        article_id = f"{article.get('title', '')}-{article.get('link', '')}"
        if article_id not in seen_articles:
            seen_articles.add(article_id)
            unique_articles.append(article)
    
    if len(unique_articles) < len(articles):
        st.info(f"ℹ️ Removed {len(articles) - len(unique_articles)} duplicate articles")
    
    for i, article in enumerate(unique_articles, 1):
        # Create a container for each article card
        with st.container():
            # Article header with title and relevance score
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Clean and format article title
                raw_title = article.get('title', 'Untitled')
                # Remove any "Pending processing:" text and clean up
                title = raw_title.replace('Pending processing:', '').strip()
                
                # Parse title to separate actual title from description/reading info
                # Look for patterns like "Title Description...6 min read3"
                import re
                # Remove reading time patterns like "6 min read" or "3 min read"
                title = re.sub(r'\d+\s*min\s*read\d*', '', title)
                # Remove standalone numbers at the end
                title = re.sub(r'\s+\d+\s*$', '', title)
                # Clean up extra spaces
                title = ' '.join(title.split())
                
                link = article.get('link', '')
                if link:
                    st.markdown(f"### [{title}]({link}) 🔗")
                else:
                    st.markdown(f"### {title}")
            
            with col2:
                # Relevance score
                score = article.get('score', 0)
                if score > 0:
                    relevance_pct = score * 100
                    if relevance_pct >= 80:
                        st.success(f"🎯 {relevance_pct:.1f}% match")
                    elif relevance_pct >= 60:
                        st.warning(f"🎯 {relevance_pct:.1f}% match")
                    else:
                        st.info(f"🎯 {relevance_pct:.1f}% match")
            
            # Article summary - clean and format properly
            raw_summary = article.get('summary', '')
            if raw_summary:
                # Remove "Pending processing:" text
                summary = raw_summary.replace('Pending processing:', '').strip()
                
                # Extract actual article description from email format
                # The summary often contains: "Title Description...6 min read3 🔗"
                # We want just the meaningful description part
                
                # Remove the title from summary if it's repeated
                if title and title in summary:
                    summary = summary.replace(title, '').strip()
                
                # Remove reading time info and links
                summary = re.sub(r'\d+\s*min\s*read\d*', '', summary)
                summary = re.sub(r'🔗\s*$', '', summary)
                summary = re.sub(r'\s+\d+\s*🔗\s*$', '', summary)
                summary = re.sub(r'\s+\d+\s*$', '', summary)
                
                # Clean up and format
                summary = ' '.join(summary.split())
                
                # Truncate very long summaries for better readability
                if len(summary) > 400:
                    summary = summary[:400] + "..."
                
                if summary:  # Only show if there's meaningful content
                    st.markdown(f"**📖 Summary:** {summary}")
            
            # Metadata row
            meta_cols = st.columns([2, 2, 2, 2])
            
            with meta_cols[0]:
                # Author information (remove source since it's implicit)
                if 'author' in article and article['author']:
                    st.markdown(f"**👤 Author:** {article['author']}")
            
            with meta_cols[1]:
                # Processing date
                processed_at = article.get('processed_at', '')
                if processed_at:
                    try:
                        # Parse and format the date
                        dt = datetime.fromisoformat(processed_at.replace('Z', '+00:00'))
                        formatted_date = dt.strftime('%b %d, %Y')
                        st.markdown(f"**📅 Indexed:** {formatted_date}")
                    except:
                        st.markdown(f"**📅 Indexed:** {processed_at[:10]}")
            
            with meta_cols[2]:
                # AI Analysis data if available
                ai_analysis = article.get('ai_analysis', {})
                if ai_analysis:
                    quality_score = ai_analysis.get('quality_score', 0)
                    if quality_score > 0:
                        st.markdown(f"**⭐ Quality:** {quality_score*100:.0f}%")
                    
                    category = ai_analysis.get('primary_category', '')
                    if category:
                        st.markdown(f"**📂 Category:** {category}")
                else:
                    # Show query that matched if available
                    query_matched = article.get('query_matched', '')
                    if query_matched:
                        st.markdown(f"**🔍 Matched:** \"{query_matched}\"")
            
            with meta_cols[3]:
                # Additional AI insights
                if ai_analysis:
                    read_time = ai_analysis.get('estimated_read_time', 0)
                    if read_time > 0:
                        st.markdown(f"**⏱️ Read time:** {read_time} min")
                    
                    difficulty = ai_analysis.get('difficulty_level', '')
                    if difficulty:
                        difficulty_emoji = {
                            'beginner': '🟢',
                            'intermediate': '🟡', 
                            'advanced': '🟠',
                            'expert': '🔴'
                        }.get(difficulty, '⚪')
                        st.markdown(f"**📈 Level:** {difficulty_emoji} {difficulty.title()}")
                
                # Technologies mentioned
                if ai_analysis and ai_analysis.get('technologies_mentioned'):
                    techs = ai_analysis['technologies_mentioned'][:2]  # Show top 2
                    if techs:
                        st.markdown(f"**🛠️ Tech:** {', '.join(techs)}")
            
            # Action buttons
            button_cols = st.columns([1, 1, 4])
            
            with button_cols[0]:
                if link:
                    st.link_button("📖 Read Article", link)
            
            with button_cols[1]:
                # Copy link button
                if link:
                    if st.button(f"📋 Copy", key=f"copy_{i}", help="Copy article link"):
                        st.code(link)
        
        # Divider between articles
        st.markdown("---")
    
    # Display AI analysis summary if available
    if any('ai_analysis' in article for article in articles):
        display_ai_analysis_summary(articles)

def display_ai_analysis_summary(articles: list):
    """
    Display a summary of AI analysis results across all articles.
    
    Args:
        articles: List of articles with AI analysis data
    """
    st.markdown("---")
    st.subheader("🧠 AI Analysis Summary")
    
    # Collect AI analysis data
    ai_articles = [article for article in articles if 'ai_analysis' in article]
    
    if not ai_articles:
        st.info("No AI analysis data available for these articles.")
        return
    
    # Calculate aggregated metrics
    total_articles = len(ai_articles)
    avg_quality = sum(article['ai_analysis'].get('quality_score', 0) for article in ai_articles) / total_articles
    avg_confidence = sum(article['ai_analysis'].get('confidence_score', 0) for article in ai_articles) / total_articles
    
    # Count categories and types
    categories = {}
    content_types = {}
    difficulty_levels = {}
    technologies = {}
    
    for article in ai_articles:
        ai_data = article['ai_analysis']
        
        # Categories
        category = ai_data.get('primary_category', 'Unknown')
        categories[category] = categories.get(category, 0) + 1
        
        # Content types
        content_type = ai_data.get('content_type', 'unknown')
        content_types[content_type] = content_types.get(content_type, 0) + 1
        
        # Difficulty levels
        difficulty = ai_data.get('difficulty_level', 'unknown')
        difficulty_levels[difficulty] = difficulty_levels.get(difficulty, 0) + 1
        
        # Technologies
        for tech in ai_data.get('technologies_mentioned', []):
            technologies[tech] = technologies.get(tech, 0) + 1
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 AI Analyzed Articles", total_articles)
    
    with col2:
        st.metric("⭐ Avg Quality Score", f"{avg_quality:.1%}")
    
    with col3:
        st.metric("🎯 Avg Confidence", f"{avg_confidence:.1%}")
    
    with col4:
        high_quality_count = sum(1 for article in ai_articles 
                               if article['ai_analysis'].get('quality_score', 0) >= 0.8)
        st.metric("🌟 High Quality (80%+)", high_quality_count)
    
    # Display distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📂 Content Categories**")
        if categories:
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
                percentage = (count / total_articles) * 100
                st.write(f"• {category}: {count} ({percentage:.1f}%)")
        
        st.markdown("**📈 Difficulty Levels**")
        if difficulty_levels:
            difficulty_order = ['beginner', 'intermediate', 'advanced', 'expert']
            for level in difficulty_order:
                if level in difficulty_levels:
                    count = difficulty_levels[level]
                    percentage = (count / total_articles) * 100
                    emoji = {'beginner': '🟢', 'intermediate': '🟡', 'advanced': '🟠', 'expert': '🔴'}.get(level, '⚪')
                    st.write(f"• {emoji} {level.title()}: {count} ({percentage:.1f}%)")
    
    with col2:
        st.markdown("**📝 Content Types**")
        if content_types:
            type_emojis = {
                'tutorial': '📚', 'opinion': '💭', 'news': '📰', 
                'research': '🔬', 'case_study': '📊', 'article': '📄'
            }
            for content_type, count in sorted(content_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_articles) * 100
                emoji = type_emojis.get(content_type, '📄')
                st.write(f"• {emoji} {content_type.title()}: {count} ({percentage:.1f}%)")
        
        st.markdown("**🛠️ Popular Technologies**")
        if technologies:
            for tech, count in sorted(technologies.items(), key=lambda x: x[1], reverse=True)[:5]:
                percentage = (count / total_articles) * 100
                st.write(f"• {tech}: {count} ({percentage:.1f}%)")
    
    # Quality distribution
    st.markdown("**⭐ Quality Score Distribution**")
    quality_ranges = {'90-100%': 0, '80-89%': 0, '70-79%': 0, '60-69%': 0, 'Below 60%': 0}
    
    for article in ai_articles:
        score = article['ai_analysis'].get('quality_score', 0) * 100
        if score >= 90:
            quality_ranges['90-100%'] += 1
        elif score >= 80:
            quality_ranges['80-89%'] += 1
        elif score >= 70:
            quality_ranges['70-79%'] += 1
        elif score >= 60:
            quality_ranges['60-69%'] += 1
        else:
            quality_ranges['Below 60%'] += 1
    
    quality_cols = st.columns(5)
    for i, (range_name, count) in enumerate(quality_ranges.items()):
        with quality_cols[i]:
            percentage = (count / total_articles) * 100 if total_articles > 0 else 0
            color = ['🔴', '🟠', '🟡', '🟢', '🟢'][i] if count > 0 else '⚪'
            st.metric(f"{color} {range_name}", f"{count} ({percentage:.0f}%)")

def show_system_status():
    """
    Display agent system status and statistics.
    """
    st.subheader("🚀 Event-Driven Service Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Health check
        health_data = call_backend_api("/health")
        if health_data:
            if health_data.get("status") == "healthy":
                st.success("✅ System Healthy")
                st.metric("🤖 Agents", health_data.get("agents_count", 0))
                st.metric("⚙️ Background Tasks", health_data.get("background_tasks", 0))
            else:
                st.error("❌ System Unhealthy")
        else:
            st.error("❌ Cannot connect to backend")
    
    with col2:
        # System status
        system_status = call_backend_api("/status")
        if system_status:
            service_status = system_status.get("service_status", "unknown")
            if service_status == "running":
                st.success("🚀 Event-Driven Multi-Agent Service: Running")
            else:
                st.error("❌ Service: Stopped")
            
            architecture = system_status.get("architecture", "unknown")
            st.write(f"**🔧 Service Type:** {architecture.replace('_', ' ').title()}")
            
            current_operation = system_status.get("current_operation", {})
            if current_operation and current_operation.get("status") != "idle":
                op_status = current_operation.get("status", "unknown")
                st.write(f"**⚙️ Current Operation:** {op_status.title()}")
                if "message" in current_operation:
                    st.write(f"• {current_operation['message']}")
            else:
                st.write("**⚙️ Current Operation:** Idle")
        else:
            st.error("❌ Cannot get system status")

def background_scheduler_section():
    """
    Section showing background scheduler status.
    """
    st.subheader("🔄 Background Scheduler")
    
    # Get system status to show scheduler info
    system_status = call_backend_api("/status")
    scheduler_info = None
    if system_status:
        scheduler_info = system_status.get("coordinator", {}).get("background_scheduler", {})
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if scheduler_info:
            scheduler_running = scheduler_info.get("running", False)
            last_fetch = scheduler_info.get("last_fetch_time")
            next_fetch_hours = scheduler_info.get("next_fetch_in_hours", 0)
            fetch_interval = scheduler_info.get("fetch_interval_hours", 24)
            
            if scheduler_running:
                st.success("✅ Scheduler Running")
            else:
                st.error("❌ Scheduler Stopped")
            
            st.metric("🕐 Fetch Interval", f"{fetch_interval} hours")
            
            if last_fetch:
                from datetime import datetime
                last_fetch_dt = datetime.fromisoformat(last_fetch.replace('Z', '+00:00'))
                st.write(f"**Last Fetch:** {last_fetch_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.write("**Last Fetch:** Never")
            
            if next_fetch_hours > 0:
                st.write(f"**Next Fetch:** In {next_fetch_hours:.1f} hours")
            else:
                st.write("**Next Fetch:** Scheduled soon")
        else:
            st.error("❌ Cannot get scheduler status")
        
        # Test Backend connection
        if st.button("🧪 Test Backend Connection", key="test_backend"):
            with st.spinner("Testing backend connection..."):
                response = call_backend_api("/health")
                
                if response and response.get("status") == "healthy":
                    st.success("✅ Backend connection successful")
                    st.info(f"Service: {response.get('service', 'unknown')}")
                else:
                    st.error("❌ Backend connection failed")
        
        # Medium cookies configuration
        st.markdown("---")
        st.subheader("🍪 Medium Member Access")
        
        with st.expander("Configure Medium Cookies for Member Content", expanded=False):
            st.markdown("""
            **To access Medium member-only stories, provide your session cookies:**
            
            1. Go to [medium.com](https://medium.com) and log in
            2. Press F12 to open Developer Tools
            3. Go to Application/Storage → Cookies → https://medium.com
            4. Copy the `sid` cookie value (required)
            5. Copy the `uid` cookie value (optional but recommended)
            """)
            
            with st.form("medium_cookies_form"):
                medium_sid = st.text_input(
                    "Medium SID Cookie *", 
                    type="password",
                    help="Required - Your Medium session ID cookie"
                )
                medium_uid = st.text_input(
                    "Medium UID Cookie", 
                    type="password",
                    help="Optional - Your Medium user ID cookie"
                )
                
                submitted = st.form_submit_button("🔐 Configure Medium Access")
                
                if submitted:
                    if medium_sid:
                        cookie_data = {"medium_sid": medium_sid}
                        if medium_uid:
                            cookie_data["medium_uid"] = medium_uid
                        
                        st.info("💡 Medium cookies are configured via the credentials folder. Please ensure your medium_cookies.json file is properly set up.")
                    else:
                        st.warning("Please provide at least the SID cookie")
            
            st.markdown("### 📋 Detailed Instructions:")
            st.write("• Go to medium.com and log in to your account")
            st.write("• Open browser developer tools (F12)")
            st.write("• Navigate to Application/Storage → Cookies → https://medium.com")
            st.write("• Copy the 'sid' cookie value (long alphanumeric string)")
            st.write("• Copy the 'uid' cookie value (shorter string)")
            st.write("• Update credentials/medium_cookies.json with these values")
            st.write("• Restart the backend to apply changes")
    
    with col2:
        # Show fetch status if operation is running
        if st.session_state.get("fetch_started", False):
            st.write("**Fetch Status:**")
            
            status_data = call_backend_api("/fetch-status")
            if status_data:
                status = status_data.get("status", "unknown")
                message = status_data.get("message", "")
                progress = status_data.get("progress", 0)
                
                # Progress bar
                st.progress(progress / 100)
                
                # Status message
                if status == "running":
                    st.info(f"🔄 {message}")
                elif status == "completed":
                    st.success(f"✅ {message}")
                    articles_processed = status_data.get("articles_processed", 0)
                    if articles_processed > 0:
                        st.metric("Articles Processed", articles_processed)
                    st.session_state.fetch_started = False
                elif status == "error":
                    st.error(f"❌ {message}")
                    st.session_state.fetch_started = False
                else:
                    st.write(f"Status: {status}")
                    st.write(f"Message: {message}")
            
            # Auto-refresh disabled to prevent recursion issues
            # Use the "Refresh Now" button in the Fetch Status tab for manual updates

def search_articles_section():
    """
    Section for searching articles with smart fetch detection.
    """
    st.subheader("🔍 Search Articles")
    
    # Check if database has articles and if fetch is needed
    stats_data = call_backend_api("/stats/realtime")
    total_articles = stats_data.get("total_articles", 0) if stats_data else 0
    
    # Smart fetch prompt
    if total_articles == 0:
        st.warning("⚠️ No articles found in database. You need to fetch articles first.")
        if st.button("🚀 Start Fetching Articles", type="primary"):
            response = call_backend_api("/fetch", "POST")
            if response and response.get("success"):
                st.success("✅ Fetch started! Check the 'Fetch Status' tab to monitor progress.")
                st.info("💡 You can search once the fetch is complete.")
                st.session_state.fetch_started = True
                # Auto-refresh disabled to prevent recursion
            else:
                st.error("❌ Failed to start fetch operation")
        return
    
    # Check if fetch is in progress
    fetch_status = call_backend_api("/fetch-status")
    if fetch_status and fetch_status.get("status") == "running":
        st.info("🔄 Articles are currently being fetched. Please wait for completion or check 'Fetch Status' tab.")
        return
    
    # Search form
    with st.form("search_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Enter your search query:",
                placeholder="e.g., machine learning, python, data science, Venkat Kaza",
                help="Use natural language to describe what you're looking for"
            )
        
        with col2:
            top_k = st.number_input(
                "Number of results:",
                min_value=1,
                max_value=50,
                value=10,
                help="Maximum number of articles to return"
            )
        
        search_submitted = st.form_submit_button("🔍 Search")
    
    # Display current database stats
    st.info(f"📊 Currently searching through {total_articles} indexed articles")
    
    # Execute search
    if search_submitted and search_query:
        with st.spinner(f"Searching for: {search_query}"):
            search_data = {
                "query": search_query,
                "top_k": top_k
            }
            
            response = call_backend_api("/search", "POST", search_data)
            
            if response:
                results = response.get("results", [])
                total_found = response.get("total_found", 0)
                
                if results:
                    st.success(f"Found {total_found} relevant articles")
                    
                    # Display results with new card layout
                    display_articles_as_cards(results)
                    
                    # Download option
                    if results:
                        df = pd.DataFrame(results)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("🔍 No articles found matching your query.")
                    
                    # Suggest fetching new articles
                    st.info("💡 Try different keywords or fetch new articles if you expect more results.")
                    if st.button("🔄 Fetch New Articles", key="search_fetch_btn"):
                        response = call_backend_api("/fetch", "POST")
                        if response and response.get("success"):
                            st.success("✅ Fetch started! Check 'Fetch Status' tab.")
                            st.session_state.fetch_started = True
            else:
                st.error("Search failed. Please try again.")
    
    elif search_submitted and not search_query:
        st.warning("Please enter a search query.")

def fetch_status_section():
    """
    Enhanced section for monitoring fetch status with terminal-like real-time updates.
    """
    st.subheader("📊 Real-time Fetch Status Monitor")
    
    # Control panel
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=True, help="Automatically refresh every 2 seconds")
    with col2:
        if st.button("⚡ Refresh Now", key="manual_refresh"):
            pass  # Manual refresh - page will reload automatically
    with col3:
        show_debug = st.checkbox("🔧 Debug Mode", help="Show additional debug information")
    with col4:
        st.info("📅 Fetching runs automatically every 24 hours")
    
    # Get current fetch status
    status_data = call_backend_api("/fetch-status")
    
    if not status_data:
        st.error("❌ Unable to fetch status from backend. Backend may be down.")
        return
    
    # Extract status information
    status = status_data.get("status", "unknown")
    message = status_data.get("message", "")
    progress = status_data.get("progress", 0)
    current_step = status_data.get("current_step", "")
    articles_processed = status_data.get("articles_processed", 0)
    articles_indexed = status_data.get("articles_indexed", 0)
    start_time = status_data.get("start_time")
    end_time = status_data.get("end_time")
    trace_messages = status_data.get("trace_messages", [])
    
    # === STATUS OVERVIEW ===
    status_emoji = {
        "idle": "⏸️",
        "running": "🔄",
        "completed": "✅",
        "error": "❌",
        "processing": "⚙️",
        "indexing": "📊"
    }.get(status, "❓")
    
    # Create status indicator with color
    status_color = {
        "idle": "🔴",
        "running": "🟡", 
        "completed": "🟢",
        "error": "🔴",
        "processing": "🟡",
        "indexing": "🟠"
    }.get(status, "⚪")
    
    st.markdown(f"## {status_emoji} {status_color} Status: **{status.upper()}**")
    
    # === LIVE PROGRESS BAR ===
    if status in ["running", "processing", "indexing"] or progress > 0:
        # Enhanced progress bar with percentage and ETA calculation
        progress_col1, progress_col2 = st.columns([4, 1])
        
        with progress_col1:
            # Color-coded progress bar
            if progress >= 90:
                st.success(f"Progress: {progress:.1f}%")
            elif progress >= 50:
                st.warning(f"Progress: {progress:.1f}%")
            else:
                st.info(f"Progress: {progress:.1f}%")
            
            st.progress(min(progress / 100, 1.0))
        
        with progress_col2:
            # Calculate ETA if we have start time and progress
            if start_time and progress > 5:
                try:
                    start_dt = datetime.fromisoformat(start_time)
                    elapsed_minutes = (datetime.now() - start_dt).total_seconds() / 60
                    if progress > 0:
                        eta_minutes = (elapsed_minutes / progress) * (100 - progress)
                        st.metric("ETA", f"{eta_minutes:.0f}m")
                    else:
                        st.metric("ETA", "∞")
                except:
                    st.metric("ETA", "--")
            else:
                st.metric("ETA", "--")
    
    # === CURRENT OPERATION ===
    if current_step:
        step_display = current_step.replace('_', ' ').replace('queue ', '').title()
        st.info(f"🔄 **Current Operation:** {step_display}")
    
    # Latest status message with highlighting
    if message:
        # Parse the message for better display
        if "📊" in message and "Processed:" in message:
            # Extract progress information from message
            try:
                if "(Processed:" in message:
                    parts = message.split("(Processed:")
                    article_name = parts[0].strip()
                    progress_part = parts[1].split(")")[0] if len(parts) > 1 else ""
                    
                    st.markdown(f"**Currently Processing:** `{article_name}`")
                    if progress_part:
                        st.caption(f"Queue Progress: {progress_part}")
                else:
                    st.write(f"**Latest:** {message}")
            except:
                st.write(f"**Latest:** {message}")
        else:
            # Display with appropriate styling
            if status == "error":
                st.error(f"❌ {message}")
            elif status == "completed":
                st.success(f"✅ {message}")
            elif "🔄" in message:
                st.info(f"🔄 {message}")
            else:
                st.write(f"**Status:** {message}")
    
    # === METRICS DASHBOARD ===
    st.markdown("### 📈 Real-time Metrics")
    
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    
    with metric_col1:
        # Calculate processing rate
        processing_rate = "--"
        if start_time and articles_processed > 0:
            try:
                start_dt = datetime.fromisoformat(start_time)
                elapsed_minutes = (datetime.now() - start_dt).total_seconds() / 60
                if elapsed_minutes > 0:
                    processing_rate = f"{articles_processed / elapsed_minutes:.1f}/min"
            except:
                pass
        
        st.metric(
            "📝 Processed", 
            articles_processed,
            help="Articles processed so far"
        )
        st.caption(f"Rate: {processing_rate}")
    
    with metric_col2:
        success_rate = "--"
        if articles_processed > 0:
            success_rate = f"{(articles_indexed / articles_processed) * 100:.1f}%"
        
        st.metric(
            "✅ Indexed", 
            articles_indexed,
            help="Articles successfully indexed"
        )
        st.caption(f"Success: {success_rate}")
    
    with metric_col3:
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time)
                elapsed = datetime.now() - start_dt
                elapsed_str = f"{elapsed.seconds // 60}m {elapsed.seconds % 60}s"
                st.metric("⏱️ Runtime", elapsed_str)
                st.caption(start_dt.strftime("%H:%M:%S"))
            except:
                st.metric("⏱️ Runtime", "--")
                st.caption("--")
        else:
            st.metric("⏱️ Runtime", "--")
            st.caption("Not started")
    
    with metric_col4:
        if end_time:
            try:
                end_dt = datetime.fromisoformat(end_time)
                st.metric("🏁 Completed", end_dt.strftime("%H:%M:%S"))
                st.caption("Finished")
            except:
                st.metric("🏁 Completed", "--")
                st.caption("--")
        else:
            completion_status = "In Progress" if status == "running" else "Not completed"
            st.metric("🏁 Completed", "--")
            st.caption(completion_status)
    
    with metric_col5:
        # Show queue remaining if available
        remaining = "--"
        if "Processed:" in message:
            try:
                # Extract total from message like "(Processed: 116/2287)"
                import re
                match = re.search(r'(\d+)/(\d+)', message)
                if match:
                    processed, total = map(int, match.groups())
                    remaining = total - processed
            except:
                pass
        
        st.metric(
            "⏳ Remaining", 
            remaining,
            help="Estimated articles remaining"
        )
        
        if remaining != "--" and remaining > 0:
            # Calculate estimated completion time
            if start_time and articles_processed > 0:
                try:
                    start_dt = datetime.fromisoformat(start_time)
                    elapsed_minutes = (datetime.now() - start_dt).total_seconds() / 60
                    rate = articles_processed / elapsed_minutes if elapsed_minutes > 0 else 0
                    eta_minutes = remaining / rate if rate > 0 else 0
                    st.caption(f"ETA: {eta_minutes:.0f}m")
                except:
                    st.caption("ETA: --")
            else:
                st.caption("ETA: --")
        else:
            st.caption("--")
    
    # === TERMINAL-LIKE ACTIVITY LOG ===
    st.markdown("### 💻 Live Activity Stream")
    
    if trace_messages:
        # Create terminal-style container
        terminal_style = """
        <style>
        .terminal-container {
            background-color: #1e1e1e;
            color: #00ff00;
            font-family: 'Courier New', monospace;
            padding: 15px;
            border-radius: 5px;
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #333;
        }
        .terminal-line {
            margin: 2px 0;
            white-space: pre-wrap;
        }
        .terminal-timestamp {
            color: #888;
        }
        .terminal-success {
            color: #00ff00;
        }
        .terminal-error {
            color: #ff4444;
        }
        .terminal-info {
            color: #4488ff;
        }
        .terminal-warning {
            color: #ffaa00;
        }
        </style>
        """
        
        st.markdown(terminal_style, unsafe_allow_html=True)
        
        # Show last 15 messages in terminal style
        terminal_content = ""
        for trace in reversed(trace_messages[-15:]):
            timestamp = trace.get("timestamp", "")
            msg = trace.get("message", "")
            
            try:
                ts_dt = datetime.fromisoformat(timestamp)
                formatted_time = ts_dt.strftime("%H:%M:%S")
            except:
                formatted_time = timestamp[:8] if len(timestamp) >= 8 else timestamp
            
            # Determine message type and color
            if "✅" in msg or "Successfully" in msg or "completed" in msg.lower():
                msg_class = "terminal-success"
                prefix = "[SUCCESS]"
            elif "❌" in msg or "Error" in msg or "Failed" in msg:
                msg_class = "terminal-error"
                prefix = "[ERROR]"
            elif "🔄" in msg or "Processing" in msg or "Fetching" in msg:
                msg_class = "terminal-info"
                prefix = "[INFO]"
            elif "⚠️" in msg or "Warning" in msg:
                msg_class = "terminal-warning"
                prefix = "[WARN]"
            else:
                msg_class = "terminal-info"
                prefix = "[INFO]"
            
            # Clean up emojis for terminal display
            clean_msg = msg.replace('📊', '').replace('🔄', '').replace('✅', '').replace('❌', '').replace('⚙️', '').strip()
            
            terminal_content += f'<div class="terminal-line"><span class="terminal-timestamp">[{formatted_time}]</span> <span class="{msg_class}">{prefix}</span> {clean_msg}</div>\n'
        
        # Display terminal
        st.markdown(
            f'<div class="terminal-container">{terminal_content}</div>',
            unsafe_allow_html=True
        )
        
        # Show condensed recent activity in normal UI style as well
        if st.checkbox("📋 Show Detailed Log", help="Show detailed activity log in regular UI style"):
            st.markdown("#### Recent Activity (Last 10 Messages)")
            
            for i, trace in enumerate(reversed(trace_messages[-10:])):
                timestamp = trace.get("timestamp", "")
                msg = trace.get("message", "")
                
                try:
                    ts_dt = datetime.fromisoformat(timestamp)
                    formatted_time = ts_dt.strftime("%H:%M:%S")
                except:
                    formatted_time = timestamp
                
                # Style based on message content
                if "✅" in msg or "Successfully" in msg:
                    st.success(f"**{formatted_time}** - {msg}")
                elif "❌" in msg or "Error" in msg or "Failed" in msg:
                    st.error(f"**{formatted_time}** - {msg}")
                elif "🔄" in msg or "Processing" in msg or "Fetching" in msg:
                    st.info(f"**{formatted_time}** - {msg}")
                else:
                    st.write(f"**{formatted_time}** - {msg}")
    else:
        st.info("🔄 No activity logged yet. Start a fetch operation to see live updates here!")
        
        # Show sample of what the terminal will look like
        sample_terminal = """
        <div style="background-color: #1e1e1e; color: #00ff00; font-family: 'Courier New', monospace; padding: 15px; border-radius: 5px; border: 1px solid #333;">
        <div>[12:30:45] [INFO] 🚀 Starting fetch operation...</div>
        <div>[12:30:46] [INFO] 📧 Fetching emails from Gmail...</div>
        <div>[12:30:47] [SUCCESS] Found 117 Medium Daily Digest emails</div>
        <div>[12:30:48] [INFO] ⚙️ Starting queue-based processing...</div>
        <div>[12:30:49] [INFO] Processing: AI in Plain English (1/2287)</div>
        </div>
        """
        st.markdown("**Preview - Live Terminal Output:**", unsafe_allow_html=True)
        st.markdown(sample_terminal, unsafe_allow_html=True)
    
    # === DEBUG INFORMATION ===
    if show_debug:
        st.markdown("---")
        st.subheader("🔧 Debug Information")
        
        debug_col1, debug_col2 = st.columns(2)
        
        with debug_col1:
            st.markdown("**Raw Status Data:**")
            st.json({
                "status": status,
                "progress": progress,
                "current_step": current_step,
                "articles_processed": articles_processed,
                "articles_indexed": articles_indexed,
                "trace_count": len(trace_messages)
            })
        
        with debug_col2:
            # Test backend connectivity
            st.markdown("**Backend Tests:**")
            
            # Test health endpoint
            health_data = call_backend_api("/health")
            if health_data and health_data.get("status") == "healthy":
                st.success("✅ Backend Health: OK")
            else:
                st.error("❌ Backend Health: Failed")
            
            # Test System Status
            if st.button("🧪 Test System Debug", key="debug_system_test"):
                debug_data = call_backend_api("/status")
                if debug_data and debug_data.get("service_status") == "running":
                    st.success(f"✅ Service Status: {debug_data.get('architecture', 'unknown')}")
                else:
                    st.error("❌ System Debug: Failed")
    
    # === DATABASE STATUS ===
    st.markdown("---")
    st.subheader("💾 Real-time Database Status")
    
    realtime_stats = call_backend_api("/stats/realtime")
    if realtime_stats:
        db_col1, db_col2, db_col3, db_col4 = st.columns(4)
        
        with db_col1:
            total_articles = realtime_stats.get("total_articles", 0)
            st.metric("📚 Total Articles", f"{total_articles:,}")
        
        with db_col2:
            last_update = realtime_stats.get("last_update", "")
            if last_update:
                try:
                    last_update_dt = datetime.fromisoformat(last_update.replace("Z", "+00:00"))
                    st.metric("📅 Last Update", last_update_dt.strftime("%m/%d %H:%M"))
                except:
                    st.metric("📅 Last Update", "Error")
            else:
                st.metric("📅 Last Update", "Never")
        
        with db_col3:
            db_info = realtime_stats.get("database_info", {})
            model_name = db_info.get("embedding_model", "N/A")
            # Truncate long model names for display
            if len(model_name) > 15:
                model_name = model_name[:12] + "..."
            st.metric("🧠 AI Model", model_name)
        
        with db_col4:
            # Show digest date range instead of growth rate
            digest_info = realtime_stats.get("digest_info", {})
            digest_days = digest_info.get("total_digest_days", 0)
            if digest_days > 0:
                st.metric("📊 Digest Days", digest_days)
            else:
                # Fallback: Calculate database growth rate if we have historical data
                growth_rate = "--"
                if articles_processed > 0 and start_time:
                    try:
                        start_dt = datetime.fromisoformat(start_time)
                        elapsed_hours = (datetime.now() - start_dt).total_seconds() / 3600
                        if elapsed_hours > 0:
                            growth_rate = f"+{articles_processed / elapsed_hours:.0f}/hr"
                    except:
                        pass
                st.metric("📈 Growth Rate", growth_rate)
    
    # === AUTO-REFRESH LOGIC ===
    # Auto-refresh completely disabled to prevent infinite recursion
    # Users can use the "Refresh Now" button for manual updates
    if auto_refresh and status in ["running", "processing", "indexing"]:
        st.info("🔄 Auto-refresh is temporarily disabled. Use the 'Refresh Now' button to update manually.")

def main():
    """
    Main Streamlit application.
    """
    # Initialize session state
    if "fetch_started" not in st.session_state:
        st.session_state.fetch_started = False
    
    # Header
    st.title("📧 Gmail Article Search Agent")
    st.markdown("""
    Search through your Medium Daily Digest articles using semantic search.
    This application fetches articles from your Gmail, indexes them in a vector database,
    and allows you to search using natural language queries.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # System status
        show_system_status()
        
        st.markdown("---")
        
        # Reset memory (for testing) - Feature disabled since endpoint doesn't exist
        # if st.button("🔄 Reset Memory", help="Reset last update time (for testing)"):
        #     response = call_backend_api("/reset-memory", "POST")
        #     if response and response.get("status") == "success":
        #         st.success("Memory reset successfully")
        #     else:
        #         st.error("Failed to reset memory")
        
        st.markdown("---")
        
        # Instructions
        st.markdown("""
        ### 📖 Instructions
        
        1. **Setup**: Ensure your Gmail credentials are configured
        2. **Fetch**: Click "Fetch New Articles" to get latest emails
        3. **Search**: Use natural language to find relevant articles
        4. **View**: Click on links to read the full articles
        
        ### 🔧 Requirements
        - Gmail API credentials
        - Medium Daily Digest emails in your inbox
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "🔄 Background Scheduler", "📊 Fetch Status"])
    
    with tab1:
        search_articles_section()
    
    with tab2:
        background_scheduler_section()
    
    with tab3:
        fetch_status_section()

if __name__ == "__main__":
    main()
