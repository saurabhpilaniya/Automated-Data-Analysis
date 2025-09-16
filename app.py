import sys
if sys.version_info >= (3, 12):
    import warnings
    warnings.warn("Python 3.12+ may have compatibility issues with some packages. "
                 "Consider using Python 3.11 for this application.")
import streamlit as st
import pandas as pd
import numpy as np
import os
from utils.data_processing import DataProcessor
from utils.visualization import ChartGenerator
from utils.insights import InsightGenerator
from utils.qa_system import QASystem
from utils.auto_ml import AutoML
import tempfile
import base64
from io import BytesIO, StringIO
import time
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import plotly.io as pio

# Page configuration
st.set_page_config(
    page_title="Auto Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
    }
    .report-section {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    /* Fix cursor for sidebar navigation */
    .stSidebar .stSelectbox label {
        cursor: pointer !important;
    }
    .stSidebar .stSelectbox div[data-baseweb="select"] {
        cursor: pointer !important;
    }
    .stSidebar .stSelectbox div[data-baseweb="select"]:hover {
        background-color: #f0f2f6;
        border-color: #1f77b4;
    }
    
    /* Style the select box to look more clickable */
    .stSidebar .stSelectbox div[data-baseweb="select"] > div:first-child {
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #ccc;
        transition: all 0.2s ease;
    }
    
    .stSidebar .stSelectbox div[data-baseweb="select"] > div:first-child:hover {
        border-color: #1f77b4;
        box-shadow: 0 0 0 1px #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize ALL session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None
if 'profile_report' not in st.session_state:
    st.session_state.profile_report = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Data Upload"
if 'generated_report' not in st.session_state:
    st.session_state.generated_report = None
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
if 'view_report' not in st.session_state:
    st.session_state.view_report = False
if 'generation_status' not in st.session_state:
    st.session_state.generation_status = "ready"
if 'view_full_report' not in st.session_state:
    st.session_state.view_full_report = False
if 'pdf_report' not in st.session_state:
    st.session_state.pdf_report = None
if 'chart_insights' not in st.session_state:
    st.session_state.chart_insights = None
if 'profile' not in st.session_state:
    st.session_state.profile = None

def generate_pdf_report_with_fallback(profile, chart_insights, df):
    """Generate PDF report with fallback for chart image generation"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    story = []
    
    try:
        # Title Section
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1
        )
        story.append(Paragraph("COMPREHENSIVE DATA ANALYSIS REPORT", title_style))
        story.append(Paragraph(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Section 1: Dataset Overview
        story.append(Paragraph("1. DATASET OVERVIEW", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        overview_data = [
            ["Total records", str(df.shape[0])],
            ["Total columns", str(df.shape[1])],
            ["Numeric columns", str(len(numeric_cols))],
            ["Categorical columns", str(len(categorical_cols))],
            ["Date/time columns", str(len(datetime_cols))]
        ]
        
        overview_table = Table(overview_data, colWidths=[2*inch, 2*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 20))
        
        # Section 2: Data Profile
        story.append(Paragraph("2. DATA PROFILE", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Add profile data to the report
        if profile and 'overview' in profile:
            # Add overview table
            overview_data = [["Metric", "Value"]]
            for col in profile['overview'].columns:
                for i, val in enumerate(profile['overview'][col]):
                    overview_data.append([col, str(val)])
            
            profile_table = Table(overview_data, colWidths=[2*inch, 2*inch])
            profile_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(profile_table)
            story.append(Spacer(1, 12))
        
        if profile and 'issues' in profile and profile['issues']:
            story.append(Paragraph("Data Quality Issues:", styles['Heading3']))
            for issue in profile['issues']:
                story.append(Paragraph(f"• {issue}", styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Section 3: Visual Analysis with Charts
        story.append(Paragraph("3. VISUAL ANALYSIS", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        total_charts = len(chart_insights)
        for i, (chart, chart_type, columns, insight) in enumerate(chart_insights):
            story.append(Paragraph(f"Chart {i+1}: {chart_type.upper()} - {', '.join(columns)}", styles['Heading3']))
            story.append(Spacer(1, 8))
            
            # Save chart as image with error handling for Kaleido
            try:
                # Try to export with lower resolution
                img_data = chart.to_image(format="png", width=600, height=400, scale=1)
                img_buffer = io.BytesIO(img_data)
                
                # Add chart image
                story.append(Image(img_buffer, width=6*inch, height=4*inch))
                story.append(Spacer(1, 12))
                
            except Exception as e:
                # If Kaleido is not available, use a placeholder
                error_msg = f"Chart image could not be generated. Please install kaleido with: pip install kaleido"
                story.append(Paragraph(error_msg, styles['Normal']))
                story.append(Spacer(1, 12))
            
            # Add insights
            story.append(Paragraph("Insights:", styles['Heading4']))
            insight_paragraphs = insight.split('\n\n')
            for para in insight_paragraphs[:3]:  # Limit to 3 paragraphs
                if para.strip():
                    story.append(Paragraph(para.strip().replace('**', ''), styles['Normal']))
            
            story.append(Spacer(1, 20))
            
            # Add page break after every 2 charts
            if (i + 1) % 2 == 0 and (i + 1) < total_charts:
                story.append(PageBreak())
        
        # Section 4: Recommendations
        story.append(Paragraph("4. RECOMMENDATIONS", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        recommendations = [
            "Review and address data quality issues",
            "Consider insights from visual analysis",
            "Use correlations for predictive modeling",
            "Monitor trends over time",
            "Validate findings with domain experts"
        ]
        
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 20))
        story.append(Paragraph("=" * 60, styles['Normal']))
        story.append(Paragraph("END OF REPORT", styles['Heading2']))
        story.append(Paragraph("=" * 60, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer.getvalue()
        
    except Exception as e:
        raise e

# App title
st.markdown('<h1 class="main-header"> Automated Data Analysis Platform</h1>', unsafe_allow_html=True)

# Button-based navigation
st.sidebar.title("Navigation")
if st.sidebar.button("Data Upload", use_container_width=True):
    st.session_state.current_page = "Data Upload"
if st.sidebar.button("Data Profiling", use_container_width=True):
    st.session_state.current_page = "Data Profiling"
if st.sidebar.button("Visualization", use_container_width=True):
    st.session_state.current_page = "Visualization"
if st.sidebar.button("Q&A System", use_container_width=True):
    st.session_state.current_page = "Q&A System"
if st.sidebar.button("AutoML", use_container_width=True):
    st.session_state.current_page = "AutoML"
if st.sidebar.button("Report Generation", use_container_width=True):
    st.session_state.current_page = "Report Generation"

# Set default page if not set
app_mode = st.session_state.current_page

# File upload
if app_mode == "Data Upload":
    st.header("Upload Your Dataset")
    
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
    
    # If new file is uploaded
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Store in session state
            st.session_state.df = df
            st.session_state.processed = False
            st.session_state.data_processor = DataProcessor(df)
            st.session_state.file_uploaded = True
            st.session_state.file_name = uploaded_file.name
            st.session_state.profile_report = None
            st.session_state.generated_report = None
            st.session_state.report_generated = False
            st.session_state.view_report = False
            
            st.success("File uploaded successfully!")
            st.write(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
            st.write(f"**File Name:** {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.error("Please make sure you're uploading a valid CSV or Excel file.")
    
    # Show file info if file is already uploaded
    elif st.session_state.file_uploaded and st.session_state.df is not None:
        st.success(f"File already loaded: {st.session_state.file_name}")
        st.write(f"**Dataset Shape:** {st.session_state.df.shape[0]} rows, {st.session_state.df.shape[1]} columns")
        
        # Quick summary instead of full preview
        with st.expander("Quick Dataset Summary"):
            st.write("**First 3 rows:**")
            st.dataframe(st.session_state.df.head(3))
            
            st.write("**Columns:**", list(st.session_state.df.columns))
            
            # Basic stats
            numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.write("**Numeric columns:**", list(numeric_cols))
        
        # Option to clear current file
        if st.button("Clear Current File & Upload New"):
            st.session_state.df = None
            st.session_state.file_uploaded = False
            st.session_state.file_name = None
            st.session_state.processed = False
            st.session_state.data_processor = None
            st.session_state.profile_report = None
            st.session_state.generated_report = None
            st.session_state.report_generated = False
            st.session_state.view_report = False
            st.rerun()
    
    
# Data Profiling
elif app_mode == "Data Profiling" and st.session_state.df is not None:
    st.header("Data Quality Report")
    
    # Generate or retrieve profile report
    if st.session_state.profile_report is None:
        with st.spinner("Analyzing data quality..."):
            st.session_state.profile_report = st.session_state.data_processor.generate_data_profile()
    
    profile_report = st.session_state.profile_report
    
    # Show data preview
    st.subheader("Data Preview")
    st.dataframe(st.session_state.df.head())
    
    # Show basic info
    st.subheader("Basic Information")
    
    # Create a string buffer for df.info()
    buffer = StringIO()
    st.session_state.df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    # Display the info
    st.text(info_str)
    
    # Show data types
    st.subheader("Data Types")
    dtype_df = pd.DataFrame({
        'Column': st.session_state.df.columns,
        'Data Type': st.session_state.df.dtypes.astype(str),
        'Non-Null Count': st.session_state.df.count().values,
        'Null Count': st.session_state.df.isnull().sum().values
    })
    st.dataframe(dtype_df)
    
    # Show the profile report
    st.subheader("Data Overview")
    st.dataframe(profile_report['overview'])
    
    st.subheader("Missing Values")
    st.dataframe(profile_report['missing_values'])
    
    st.subheader("Descriptive Statistics")
    st.dataframe(profile_report['stats'])
    
    # Show issues and recommendations
    if profile_report['issues']:
        st.subheader("Data Quality Issues")
        for issue in profile_report['issues']:
            st.warning(issue)
        
        st.subheader("Recommended Actions")
        if st.button("Apply Automated Data Cleaning"):
            with st.spinner("Cleaning data..."):
                cleaned_df = st.session_state.data_processor.auto_clean_data()
                st.session_state.df = cleaned_df
                st.session_state.processed = True
                st.session_state.data_processor = DataProcessor(cleaned_df)
                
                # Generate new profile for cleaned data
                with st.spinner("Generating cleaned data profile..."):
                    st.session_state.profile_report = st.session_state.data_processor.generate_data_profile()
                
                st.success("Data cleaning completed!")
                st.rerun()
    else:
        st.success("No significant data quality issues detected!")
        st.session_state.processed = True

elif app_mode == "Data Profiling" and st.session_state.df is None:
    st.warning("Please upload a dataset first from the 'Data Upload' section.")

# Visualization
elif app_mode == "Visualization" and st.session_state.df is not None:
    st.header("Automated Visualizations")
    
    df = st.session_state.df
    chart_gen = ChartGenerator(df)
    insight_gen = InsightGenerator(df)
    
    with st.spinner("Generating visualizations and insights..."):
        charts = chart_gen.generate_all_charts()
        
        for i, (chart, chart_type, columns) in enumerate(charts):
            st.plotly_chart(chart, use_container_width=True)
            
            # Generate insights
            insight = insight_gen.generate_insight(chart_type, columns)
            st.markdown(f"**Insight:** {insight}")
            
            st.markdown("---")

# Q&A System
elif app_mode == "Q&A System" and st.session_state.df is not None:
    st.header("Natural Language Q&A")
    
    if st.session_state.qa_system is None:
        with st.spinner("Initializing Q&A system..."):
            st.session_state.qa_system = QASystem(st.session_state.df)
    
    question = st.text_input("Ask a question about your data:", 
                            placeholder="e.g., Which branch has the highest deposits?")
    
    if question:
        with st.spinner("Processing your question..."):
            try:
                answer, chart = st.session_state.qa_system.answer_question(question)
                
                st.subheader("Answer:")
                st.write(answer)
                
                if chart:
                    st.subheader("Visualization:")
                    st.plotly_chart(chart, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")

# AutoML
elif app_mode == "AutoML" and st.session_state.df is not None:
    st.header("Automated Machine Learning")
    
    df = st.session_state.df
    auto_ml = AutoML(df)
    
    st.info("This feature will automatically train a model if your data has a clear target variable.")
    
    # Try to detect target column
    target_col = st.selectbox("Select target variable (or let AutoML detect)", 
                             [""] + list(df.columns))
    
    if st.button("Run AutoML"):
        with st.spinner("Analyzing data and training model..."):
            try:
                if target_col == "":
                    result = auto_ml.auto_detect_and_train()
                else:
                    result = auto_ml.train_model(target_col)
                
                st.subheader("AutoML Results")
                st.write(f"**Problem Type:** {result['problem_type']}")
                st.write(f"**Target Variable:** {result['target']}")
                st.write(f"**Best Model:** {result['best_model']}")
                st.write(f"**Validation Score:** {result['score']:.4f}")
                
                if 'feature_importance' in result and result['feature_importance'] is not None:
                    st.subheader("Feature Importance")
                    st.plotly_chart(result['feature_importance'], use_container_width=True)
                    
            except Exception as e:
                st.error(f"AutoML failed: {str(e)}")

# Report Generation
elif app_mode == "Report Generation" and st.session_state.df is not None:
    st.header("Generate Comprehensive PDF Report")
    
    # Initialize session state
    if 'report_status' not in st.session_state:
        st.session_state.report_status = "ready"  # ready, generating, completed, error
    
    # Create buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Report", use_container_width=True, 
                    disabled=st.session_state.report_status == "generating"):
            st.session_state.report_status = "generating"
    
    # Generate report when button is clicked
    if st.session_state.report_status == "generating":
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Generate profile (if not already done)
            status_text.text("Generating data profile...")
            progress_bar.progress(20)
            
            # Make sure we have the profile data
            if st.session_state.profile_report is None:
                data_processor = DataProcessor(st.session_state.df)
                st.session_state.profile_report = data_processor.generate_data_profile()
            
            # Step 2: Generate charts
            status_text.text("Creating charts...")
            progress_bar.progress(40)
            chart_gen = ChartGenerator(st.session_state.df)
            insight_gen = InsightGenerator(st.session_state.df)
            
            # Use the generate_all_charts method
            all_charts = chart_gen.generate_all_charts()
            
            # Include more charts in the report (up to 10 or all available)
            max_charts = min(10, len(all_charts))
            charts = []
            
            for i, (chart, chart_type, columns) in enumerate(all_charts[:max_charts]):
                insight = insight_gen.generate_insight(chart_type, columns)
                charts.append((chart, chart_type, columns, insight))
                progress_bar.progress(40 + int((i+1)*30/max_charts))
            
            # Step 3: Generate PDF with error handling for chart images
            status_text.text("Creating PDF document...")
            progress_bar.progress(80)
            
            # Generate PDF with proper error handling
            pdf_bytes = generate_pdf_report_with_fallback(
                st.session_state.profile_report, 
                charts, 
                st.session_state.df
            )
            
            # Store results
            st.session_state.pdf_data = pdf_bytes
            st.session_state.report_status = "completed"
            st.session_state.chart_insights = charts
            
            # Final step
            status_text.text("Report generated successfully!")
            progress_bar.progress(100)
            time.sleep(1)
            
        except Exception as e:
            st.session_state.report_status = "error"
            st.error(f"Error generating PDF report: {str(e)}")
    
    # Show download button if report is completed
    if st.session_state.report_status == "completed" and st.session_state.pdf_data is not None:
        st.download_button(
            label="Download PDF Report",
            data=st.session_state.pdf_data,
            file_name=f"data_analysis_report_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
        # Show preview option
        if st.button("View Report Preview", use_container_width=True):
            st.session_state.view_full_report = True
    
    # Show status message
    if st.session_state.report_status == "generating":
        st.info("PDF report generation in progress...")
    
    elif st.session_state.report_status == "error":
        st.error("Report generation failed. Please try again.")
        if st.button("Reset Report Generator"):
            st.session_state.report_status = "ready"
    
    # View complete report in the app
    if st.session_state.get('view_full_report', False) and st.session_state.get('chart_insights'):
        st.subheader("Report Preview")
        
        # Display data profile
        st.markdown("Data Profile Summary")
        if st.session_state.get('profile_report'):
            profile = st.session_state.profile_report
            st.write("**Data Overview:**")
            st.dataframe(profile['overview'])
            
            st.write("**Missing Values:**")
            st.dataframe(profile['missing_values'])
            
            if profile['issues']:
                st.write("**Data Quality Issues:**")
                for issue in profile['issues']:
                    st.warning(issue)
            else:
                st.success("No data quality issues found")
        
        # Display all charts with insights
        for i, (chart, chart_type, columns, insight) in enumerate(st.session_state.chart_insights):
            st.markdown(f"Chart {i+1}: {chart_type.title()} - {', '.join(columns)}")
            st.plotly_chart(chart, use_container_width=True)
            st.markdown(f"**Insights:**")
            st.markdown(insight)
            st.markdown("---")
    
    elif st.session_state.report_status == "ready":
        st.info("Click 'Generate Report' to create a PDF report with key insights")
# Handle cases where no data is uploaded

elif st.session_state.df is None:
    st.warning("Please upload a dataset first from the Data Upload section.")

else:
    st.info("Select a mode from the sidebar to begin analysis.")
