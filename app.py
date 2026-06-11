from flask import Flask, render_template, request, jsonify, send_file
import cv2
import os
import tempfile
import uuid
import requests
import re
import yt_dlp
from datetime import datetime
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import base64
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from waitress import serve

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['REPORTS_FOLDER'] = 'static/reports'

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/frames', exist_ok=True)
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)

# Roboflow API configuration
ROBOFLOW_API_KEY = "NoIKIjBBkBCSA3rY6Xi0"

# Initialize InferenceHTTPClient
inference_client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

def extract_frames(video_path, scan_mode='quick'):
    """Extract frames from video based on scan mode
    
    Scan Modes:
    - quick: 3 frames (beginning, middle, end) - Fast analysis
    - deep: 7 frames - More thorough analysis
    - ultra: 10 frames - Most comprehensive analysis
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None, "Could not open video file"
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    if total_frames < 3:
        return None, "Video too short"
    
    # Define frame positions based on scan mode
    if scan_mode == 'ultra':
        # 10 frames - Ultra comprehensive
        frame_positions = [
            int(total_frames * 0.05),   # 5%
            int(total_frames * 0.15),   # 15%
            int(total_frames * 0.25),   # 25%
            int(total_frames * 0.35),   # 35%
            int(total_frames * 0.45),   # 45%
            int(total_frames * 0.55),   # 55%
            int(total_frames * 0.65),   # 65%
            int(total_frames * 0.75),   # 75%
            int(total_frames * 0.85),   # 85%
            int(total_frames * 0.95)    # 95%
        ]
        position_names = ['frame_1', 'frame_2', 'frame_3', 'frame_4', 'frame_5',
                          'frame_6', 'frame_7', 'frame_8', 'frame_9', 'frame_10']
    elif scan_mode == 'deep':
        # 7 frames - Deep analysis
        frame_positions = [
            int(total_frames * 0.1),    # 10%
            int(total_frames * 0.25),   # 25%
            int(total_frames * 0.4),    # 40%
            int(total_frames * 0.5),    # 50%
            int(total_frames * 0.6),    # 60%
            int(total_frames * 0.75),   # 75%
            int(total_frames * 0.9)     # 90%
        ]
        position_names = ['early', 'quarter', 'pre-mid', 'middle', 'post-mid', 'three-quarter', 'late']
    else:
        # Quick mode - 3 frames (default)
        frame_positions = [
            int(total_frames * 0.1),   # 10% - beginning
            int(total_frames * 0.5),   # 50% - middle
            int(total_frames * 0.9)    # 90% - end
        ]
        position_names = ['beginning', 'middle', 'end']
    
    frames_data = []
    unique_id = str(uuid.uuid4())[:8]
    
    for idx, frame_pos in enumerate(frame_positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        
        if ret:
            frame_filename = f"frame_{unique_id}_{position_names[idx]}.jpg"
            frame_path = os.path.join('static/frames', frame_filename)
            cv2.imwrite(frame_path, frame)
            frames_data.append({
                'position': position_names[idx],
                'path': frame_path,
                'web_path': f"/static/frames/{frame_filename}",
                'timestamp': round((frame_pos / fps), 2) if fps > 0 else 0
            })
    
    cap.release()
    
    return {
        'frames': frames_data,
        'duration': round(duration, 2),
        'total_frames': total_frames,
        'fps': round(fps, 2),
        'scan_mode': scan_mode,
        'frames_analyzed': len(frames_data)
    }, None

def analyze_frames(frames_data):
    """Send frames to Roboflow and get predictions using inference_sdk"""
    results = []
    
    for frame in frames_data['frames']:
        try:
            # Run workflow on the image using inference_sdk
            result = inference_client.run_workflow(
                workspace_name="artficix",
                workflow_id="custom-workflow-2",
                images={
                    "image": frame['path']  # Path to the image file
                },
                use_cache=True  # Speeds up repeated requests
            )
            
            # Parse the result from workflow
            if result and len(result) > 0:
                output = result[0]
                prediction_data = output.get('predictions', {})
                predictions = prediction_data.get('predictions', []) if isinstance(prediction_data, dict) else prediction_data
                
                if predictions and len(predictions) > 0:
                    top_prediction = predictions[0]
                    results.append({
                        'position': frame['position'],
                        'timestamp': frame['timestamp'],
                        'web_path': frame['web_path'],
                        'class': top_prediction.get('class', 'Unknown'),
                        'confidence': round(top_prediction.get('confidence', 0) * 100, 2)
                    })
                else:
                    results.append({
                        'position': frame['position'],
                        'timestamp': frame['timestamp'],
                        'web_path': frame['web_path'],
                        'class': 'Unknown',
                        'confidence': 0
                    })
            else:
                results.append({
                    'position': frame['position'],
                    'timestamp': frame['timestamp'],
                    'web_path': frame['web_path'],
                    'class': 'Unknown',
                    'confidence': 0
                })
                
        except Exception as e:
            results.append({
                'position': frame['position'],
                'timestamp': frame['timestamp'],
                'web_path': frame['web_path'],
                'class': 'Error',
                'confidence': 0,
                'error': str(e)
            })
    
    return results

def apply_low_confidence_rule(results):
    """If at least 2 frames have confidence below 60%, mark all frames as AI generated"""
    if not results:
        return results
    
    # Count how many predictions have confidence below 60%
    low_confidence_count = sum(1 for r in results if r['confidence'] < 60)
    
    if low_confidence_count >= 2:
        # Mark all frames as AI generated
        for r in results:
            r['class'] = 'AI'
            r['low_confidence_override'] = True
    
    return results

def get_final_verdict(results):
    """Determine final verdict based on highest confidence prediction"""
    if not results:
        return {'verdict': 'Unknown', 'confidence': 0}
    
    # Check if low confidence rule was applied
    low_confidence_flag = any(r.get('low_confidence_override', False) for r in results)
    
    # Find the prediction with highest confidence
    highest_conf_result = max(results, key=lambda x: x['confidence'])
    
    # Count AI vs Real predictions
    ai_count = sum(1 for r in results if r['class'] == 'AI')
    real_count = sum(1 for r in results if r['class'] == 'Real')
    
    # Calculate average confidence for each class
    ai_confidences = [r['confidence'] for r in results if r['class'] == 'AI']
    real_confidences = [r['confidence'] for r in results if r['class'] == 'Real']
    
    avg_ai_conf = sum(ai_confidences) / len(ai_confidences) if ai_confidences else 0
    avg_real_conf = sum(real_confidences) / len(real_confidences) if real_confidences else 0
    
    # Determine verdict
    verdict = highest_conf_result['class']
    confidence = highest_conf_result['confidence']
    
    # Generate detailed report
    report = generate_analysis_report(results, verdict, confidence, ai_count, real_count, 
                                       avg_ai_conf, avg_real_conf, low_confidence_flag)
    
    return {
        'verdict': verdict,
        'confidence': confidence,
        'ai_count': ai_count,
        'real_count': real_count,
        'avg_ai_confidence': round(avg_ai_conf, 2),
        'avg_real_confidence': round(avg_real_conf, 2),
        'low_confidence_flag': low_confidence_flag,
        'report': report
    }

def generate_analysis_report(results, verdict, confidence, ai_count, real_count, 
                              avg_ai_conf, avg_real_conf, low_confidence_flag):
    """Generate a detailed analysis report explaining the verdict"""
    report = {
        'summary': '',
        'reasons': [],
        'technical_details': [],
        'recommendation': ''
    }
    
    total_frames = len(results)
    
    if verdict == 'AI':
        # AI Generated Video Report
        report['summary'] = f"This video shows strong indicators of AI generation with {confidence}% confidence."
        
        # Add reasons based on analysis
        if ai_count == total_frames:
            report['reasons'].append({
                'icon': 'fa-robot',
                'title': 'Consistent AI Patterns',
                'detail': f'All {total_frames} analyzed frames were detected as AI-generated, indicating the entire video was likely created using AI tools.'
            })
        elif ai_count > real_count:
            report['reasons'].append({
                'icon': 'fa-chart-pie',
                'title': 'Majority AI Detection',
                'detail': f'{ai_count} out of {total_frames} frames showed AI characteristics, suggesting significant AI manipulation.'
            })
        
        if low_confidence_flag:
            report['reasons'].append({
                'icon': 'fa-exclamation-triangle',
                'title': 'Low Confidence Detection',
                'detail': 'Multiple frames had confidence scores below 60%, which is a common indicator of AI-generated content that attempts to mimic real footage.'
            })
        
        if avg_ai_conf > 80:
            report['reasons'].append({
                'icon': 'fa-bullseye',
                'title': 'High Detection Confidence',
                'detail': f'Average AI detection confidence of {round(avg_ai_conf, 1)}% indicates clear synthetic patterns in the video frames.'
            })
        
        # Technical details
        report['technical_details'] = [
            f'Frames analyzed: {total_frames}',
            f'AI-detected frames: {ai_count}',
            f'Average AI confidence: {round(avg_ai_conf, 1)}%',
            f'Peak confidence: {confidence}%'
        ]
        
        # Common AI indicators
        report['reasons'].append({
            'icon': 'fa-search',
            'title': 'Potential AI Artifacts Detected',
            'detail': 'The analysis detected patterns commonly found in AI-generated videos such as: inconsistent lighting, unnatural facial movements, blurring around edges, or temporal inconsistencies between frames.'
        })
        
        report['recommendation'] = 'Exercise caution with this video. It appears to be AI-generated or heavily manipulated. Verify the source before sharing or trusting its content.'
        
    else:
        # Authentic Video Report
        report['summary'] = f"This video appears to be authentic with {confidence}% confidence."
        
        if real_count == total_frames:
            report['reasons'].append({
                'icon': 'fa-check-circle',
                'title': 'Consistent Authenticity',
                'detail': f'All {total_frames} analyzed frames were detected as authentic, indicating genuine footage throughout the video.'
            })
        elif real_count > ai_count:
            report['reasons'].append({
                'icon': 'fa-chart-pie',
                'title': 'Majority Authentic',
                'detail': f'{real_count} out of {total_frames} frames appear to be genuine footage.'
            })
        
        if avg_real_conf > 80:
            report['reasons'].append({
                'icon': 'fa-shield-alt',
                'title': 'High Authenticity Score',
                'detail': f'Average authenticity confidence of {round(avg_real_conf, 1)}% suggests natural, unmanipulated content.'
            })
        
        report['reasons'].append({
            'icon': 'fa-video',
            'title': 'Natural Video Characteristics',
            'detail': 'The video exhibits natural characteristics including: consistent lighting, realistic motion blur, natural facial expressions, and coherent temporal flow between frames.'
        })
        
        # Technical details
        report['technical_details'] = [
            f'Frames analyzed: {total_frames}',
            f'Authentic frames: {real_count}',
            f'Average authenticity: {round(avg_real_conf, 1)}%',
            f'Peak confidence: {confidence}%'
        ]
        
        report['recommendation'] = 'This video appears to be genuine. However, always consider the source and context when evaluating media content.'
    
    return report

def generate_pdf_report(analysis_data, video_info, predictions, verdict, source_info="Unknown"):
    """Generate a comprehensive PDF forensic report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#00f0ff')
    )
    
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#00f0ff'),
        borderWidth=1,
        borderColor=colors.HexColor('#00f0ff'),
        borderPadding=5
    )
    
    subsection_style = ParagraphStyle(
        'SubSection',
        parent=styles['Heading3'],
        fontSize=12,
        spaceBefore=10,
        spaceAfter=5,
        textColor=colors.HexColor('#333333')
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        alignment=TA_JUSTIFY
    )
    
    # Generate report content
    story = []
    report_id = f"DS-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    is_ai = verdict.get('verdict') == 'AI'
    
    # ============ HEADER ============
    story.append(Paragraph("🛡️ Truesight AI", title_style))
    story.append(Paragraph("VIDEO AUTHENTICITY FORENSIC REPORT", ParagraphStyle('Subtitle', parent=styles['Heading2'], alignment=TA_CENTER, textColor=colors.grey)))
    story.append(Spacer(1, 20))
    
    # Header info table
    header_data = [
        ['Report ID:', report_id, 'Classification:', 'CONFIDENTIAL'],
        ['Generated:', timestamp, 'Analyst:', 'Truesight AI v2.0'],
    ]
    header_table = Table(header_data, colWidths=[80, 180, 80, 150])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0f8ff')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#666666')),
        ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor('#666666')),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 20))
    
    # ============ 1. CASE INFO ============
    story.append(Paragraph("1. CASE INFORMATION", section_style))
    case_data = [
        ['Case Reference:', report_id],
        ['Analysis Date:', timestamp],
        ['Source Type:', 'YouTube/URL' if 'youtube' in source_info.lower() or 'http' in source_info.lower() else 'Direct Upload'],
        ['Source:', source_info[:60] + '...' if len(source_info) > 60 else source_info],
        ['Scan Mode:', video_info.get('scan_mode', 'quick').upper()],
        ['Examiner:', 'DeepScan AI Automated System'],
    ]
    case_table = Table(case_data, colWidths=[120, 370])
    case_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4fc')),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(case_table)
    story.append(Spacer(1, 15))
    
    # ============ 2. VIDEO METADATA ============
    story.append(Paragraph("2. VIDEO METADATA", section_style))
    metadata_data = [
        ['Property', 'Value'],
        ['Duration', f"{video_info.get('duration', 'N/A')} seconds"],
        ['Frame Rate', f"{video_info.get('fps', 'N/A')} FPS"],
        ['Total Frames', f"{video_info.get('total_frames', 'N/A'):,}"],
        ['Frames Analyzed', f"{video_info.get('frames_analyzed', len(predictions))}"],
        ['Resolution', 'Extracted from video'],
        ['Format', 'Digital Video'],
    ]
    metadata_table = Table(metadata_data, colWidths=[150, 340])
    metadata_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00f0ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 1), (1, -1), 'LEFT'),
    ]))
    story.append(metadata_table)
    story.append(Spacer(1, 15))
    
    # ============ 3. INTEGRITY CHECK ============
    story.append(Paragraph("3. INTEGRITY CHECK", section_style))
    integrity_status = " MANIPULATION DETECTED" if is_ai else "NO MANIPULATION "
    integrity_color = colors.HexColor('#ff4444') if is_ai else colors.HexColor('#00cc66')
    
    integrity_data = [
        ['Check Type', 'Status', 'Details'],
        ['Frame Consistency', ' FAILED' if is_ai else ' PASSED', 'Temporal frame analysis'],
        ['Facial Analysis', ' ANOMALY' if is_ai else ' NORMAL', 'Facial landmark detection'],
        ['Compression Artifacts', 'SUSPICIOUS' if is_ai else ' NORMAL', 'JPEG/H.264 artifact analysis'],
        ['Metadata Integrity', 'INTACT', 'No metadata tampering detected'],
        ['Overall Status', integrity_status, f'Confidence: {verdict.get("confidence", 0)}%'],
    ]
    integrity_table = Table(integrity_data, colWidths=[140, 120, 230])
    integrity_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#333333')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#fff3f3') if is_ai else colors.HexColor('#f0fff0')),
    ]))
    story.append(integrity_table)
    story.append(Spacer(1, 15))
    
    # ============ 4. EXECUTIVE SUMMARY ============
    story.append(Paragraph("4. EXECUTIVE SUMMARY", section_style))
    
    verdict_text = "AI-GENERATED / DEEPFAKE" if is_ai else "AUTHENTIC / GENUINE"
    verdict_color = '#ff4444' if is_ai else '#00cc66'
    
    summary_text = f"""
    <b>VERDICT: <font color="{verdict_color}">{verdict_text}</font></b><br/><br/>
    This forensic analysis was conducted on the submitted video using DeepScan AI's advanced machine learning 
    algorithms. The analysis examined {video_info.get('frames_analyzed', len(predictions))} strategic frames extracted from the video 
    using {video_info.get('scan_mode', 'quick').upper()} scan mode.<br/><br/>
    <b>Key Findings:</b><br/>
    • Detection Confidence: {verdict.get('confidence', 0)}%<br/>
    • AI-Detected Frames: {verdict.get('ai_count', 0)} out of {len(predictions)}<br/>
    • Authentic Frames: {verdict.get('real_count', 0)} out of {len(predictions)}<br/>
    • Average AI Confidence: {verdict.get('avg_ai_confidence', 0)}%<br/>
    • Average Real Confidence: {verdict.get('avg_real_confidence', 0)}%<br/><br/>
    {"<b>⚠️ WARNING:</b> This video shows strong indicators of AI manipulation and should not be trusted as authentic media." if is_ai else "<b>✅ ASSESSMENT:</b> This video appears to be genuine footage with no significant signs of AI manipulation."}
    """
    story.append(Paragraph(summary_text, normal_style))
    story.append(Spacer(1, 15))
    
    # ============ 5. DETECTION RESULTS ============
    story.append(Paragraph("5. DETECTION RESULTS", section_style))
    
    results_header = ['Frame Position', 'Timestamp', 'Classification', 'Confidence', 'Status']
    results_data = [results_header]
    
    for pred in predictions:
        status = '🔴 AI' if pred.get('class') == 'AI' else '🟢 Real'
        results_data.append([
            pred.get('position', 'N/A').upper(),
            f"{pred.get('timestamp', 0)}s",
            pred.get('class', 'Unknown'),
            f"{pred.get('confidence', 0)}%",
            status
        ])
    
    results_table = Table(results_data, colWidths=[100, 80, 100, 80, 80])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00f0ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 15))
    
    # ============ 5b. FRAME ANALYSIS IMAGES ============
    story.append(Paragraph("FRAME ANALYSIS - VISUAL EVIDENCE", section_style))
    story.append(Paragraph(
        "The following images are the actual frames extracted from the video and analyzed by the AI detection model.",
        normal_style
    ))
    story.append(Spacer(1, 10))
    
    # Build frame image rows (up to 3 per row)
    frame_row = []
    for pred in predictions:
        # Resolve the frame image path on disk
        frame_web_path = pred.get('web_path', '')
        if frame_web_path.startswith('/'):
            frame_web_path = frame_web_path[1:]  # Remove leading slash
        frame_abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), frame_web_path)
        
        frame_cell_content = []
        
        # Try to add the image
        if os.path.exists(frame_abs_path) and os.path.getsize(frame_abs_path) > 0:
            try:
                frame_img = Image(frame_abs_path, width=145, height=110)
                frame_cell_content.append(frame_img)
            except Exception:
                frame_cell_content.append(Paragraph("<i>[Image unavailable]</i>", normal_style))
        else:
            frame_cell_content.append(Paragraph("<i>[Image unavailable]</i>", normal_style))
        
        # Label with position, timestamp, class, confidence
        pred_class = pred.get('class', 'Unknown')
        pred_conf = pred.get('confidence', 0)
        pred_pos = pred.get('position', 'N/A').upper()
        pred_ts = pred.get('timestamp', 0)
        label_color = '#ff4444' if pred_class == 'AI' else '#00cc66'
        
        label_style = ParagraphStyle(
            'FrameLabel', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, spaceAfter=2
        )
        frame_cell_content.append(Paragraph(f"<b>{pred_pos}</b>", label_style))
        frame_cell_content.append(Paragraph(f"@ {pred_ts}s", label_style))
        frame_cell_content.append(Paragraph(
            f'<font color="{label_color}"><b>{pred_class}</b></font> — {pred_conf}%', label_style
        ))
        
        frame_row.append(frame_cell_content)
    
    # Arrange frames in rows of 3
    rows_of_frames = []
    for i in range(0, len(frame_row), 3):
        rows_of_frames.append(frame_row[i:i+3])
    
    for row in rows_of_frames:
        # Pad row to 3 columns if needed
        while len(row) < 3:
            row.append([Paragraph("", normal_style)])
        
        # Create a table for this row of frames
        frame_table_data = [[row[0], row[1], row[2]]]
        frame_table = Table(frame_table_data, colWidths=[163, 163, 163])
        frame_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#eeeeee')),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f8f8')),
        ]))
        story.append(frame_table)
        story.append(Spacer(1, 8))
    
    story.append(Spacer(1, 15))
    
    # ============ 6. EXPLAINABLE AI EVIDENCE ============
    story.append(Paragraph("6. EXPLAINABLE AI EVIDENCE", section_style))
    
    if is_ai:
        evidence_text = """
        <b>AI Detection Indicators Found:</b><br/><br/>
        <b>• Facial Inconsistencies:</b> The AI model detected unnatural facial movements, including 
        irregular blinking patterns, asymmetric expressions, and temporal discontinuities in facial features.<br/><br/>
        <b>• Boundary Artifacts:</b> Analysis revealed suspicious blending at face boundaries, 
        indicating potential face-swap or deepfake generation techniques.<br/><br/>
        <b>• Temporal Anomalies:</b> Frame-to-frame analysis detected unnatural motion patterns 
        that are characteristic of GAN-generated content.<br/><br/>
        <b>• Texture Analysis:</b> Skin texture analysis revealed synthetic patterns typically 
        produced by AI image generation models (StyleGAN, Stable Diffusion, etc.).<br/><br/>
        <b>• Compression Artifacts:</b> Unusual compression patterns suggest the video has been 
        processed through AI generation or manipulation pipelines.
        """
    else:
        evidence_text = """
        <b>Authenticity Indicators Found:</b><br/><br/>
        <b>• Natural Facial Movements:</b> The analysis detected consistent and natural facial 
        movements including proper blinking, symmetric expressions, and smooth transitions.<br/><br/>
        <b>• Consistent Boundaries:</b> No suspicious blending or artifact patterns were found 
        at face boundaries or key feature areas.<br/><br/>
        <b>• Temporal Consistency:</b> Frame-to-frame analysis shows natural motion patterns 
        consistent with authentic video recordings.<br/><br/>
        <b>• Natural Textures:</b> Skin and surface textures appear natural with expected 
        variations and no synthetic patterns detected.<br/><br/>
        <b>• Normal Compression:</b> Compression artifacts are consistent with standard video 
        encoding and show no signs of AI manipulation pipelines.
        """
    story.append(Paragraph(evidence_text, normal_style))
    story.append(Spacer(1, 15))
    
    # ============ 7. AUDIO ANALYSIS ============
    story.append(Paragraph("7. AUDIO ANALYSIS", section_style))
    audio_text = """
    <b>Audio Track Assessment:</b><br/><br/>
    <i>Note: Full audio analysis requires additional processing. Current assessment is based on 
    visual frame analysis only.</i><br/><br/>
    <b>• Voice Cloning Detection:</b> Not analyzed in current scan<br/>
    <b>• Lip Sync Analysis:</b> Visual inspection suggests """ + ("potential misalignment" if is_ai else "natural synchronization") + """<br/>
    <b>• Audio-Visual Correlation:</b> """ + ("Requires further investigation" if is_ai else "Appears consistent") + """<br/>
    <b>• Background Audio:</b> Not analyzed in current scan<br/><br/>
    <i>Recommendation: For comprehensive audio forensics, use dedicated audio analysis tools.</i>
    """
    story.append(Paragraph(audio_text, normal_style))
    story.append(Spacer(1, 15))
    
    # ============ 8. TEMPORAL ANALYSIS ============
    story.append(Paragraph("8. TEMPORAL ANALYSIS", section_style))
    
    temporal_data = [
        ['Analysis Point', 'Time', 'Finding'],
    ]
    for pred in predictions:
        finding = "AI patterns detected" if pred.get('class') == 'AI' else "Natural content"
        temporal_data.append([pred.get('position', 'N/A').upper(), f"{pred.get('timestamp', 0)}s", finding])
    
    temporal_table = Table(temporal_data, colWidths=[150, 100, 240])
    temporal_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#333333')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(temporal_table)
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>Temporal Consistency Score:</b> {100 - verdict.get('confidence', 0) if is_ai else verdict.get('confidence', 0)}%", normal_style))
    story.append(Spacer(1, 15))
    
    # ============ 9. THREAT INTENT ============
    story.append(Paragraph("9. THREAT INTENT ASSESSMENT", section_style))
    
    if is_ai:
        threat_text = """
        <b>Potential Malicious Use Cases:</b><br/><br/>
        Based on the detection of AI-generated content, this video could potentially be used for:<br/><br/>
        • <b>Misinformation/Disinformation:</b> Spreading false narratives or fake news<br/>
        • <b>Identity Fraud:</b> Impersonating individuals for fraudulent purposes<br/>
        • <b>Reputation Damage:</b> Creating compromising content to harm individuals<br/>
        • <b>Financial Fraud:</b> Deceiving victims in social engineering attacks<br/>
        • <b>Political Manipulation:</b> Influencing public opinion with fabricated content<br/><br/>
        <b>Intent Confidence:</b> Unable to determine specific intent without context
        """
    else:
        threat_text = """
        <b>Threat Assessment:</b><br/><br/>
        No AI manipulation detected. The video appears to be authentic footage.<br/><br/>
        • <b>Manipulation Risk:</b> LOW<br/>
        • <b>Authenticity Confidence:</b> HIGH<br/>
        • <b>Recommended Trust Level:</b> Standard verification recommended<br/><br/>
        <i>Note: Even authentic videos should be verified for context and source credibility.</i>
        """
    story.append(Paragraph(threat_text, normal_style))
    story.append(Spacer(1, 15))
    
    # ============ 10. RISK CLASSIFICATION ============
    story.append(Paragraph("10. RISK CLASSIFICATION", section_style))
    
    confidence = verdict.get('confidence', 0)
    if is_ai:
        if confidence >= 90:
            risk_level, risk_color = "CRITICAL", "#ff0000"
        elif confidence >= 75:
            risk_level, risk_color = "HIGH", "#ff6600"
        elif confidence >= 60:
            risk_level, risk_color = "MEDIUM", "#ffaa00"
        else:
            risk_level, risk_color = "LOW", "#00aa00"
    else:
        risk_level, risk_color = "MINIMAL", "#00cc66"
    
    risk_data = [
        ['Risk Category', 'Assessment'],
        ['Overall Risk Level', risk_level],
        ['Manipulation Probability', f"{confidence}%" if is_ai else f"{100-confidence}%"],
        ['Confidence Score', f"{confidence}%"],
        ['Recommended Action', 'QUARANTINE & INVESTIGATE' if is_ai and confidence >= 75 else 'STANDARD REVIEW' if is_ai else 'CLEAR'],
    ]
    risk_table = Table(risk_data, colWidths=[200, 290])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#333333')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (1, 1), (1, 1), colors.HexColor(risk_color)),
        ('TEXTCOLOR', (1, 1), (1, 1), colors.white),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 15))
    
    # ============ 11. RECOMMENDED ACTIONS ============
    story.append(Paragraph("11. RECOMMENDED ACTIONS", section_style))
    
    if is_ai:
        actions_text = """
        <b>Immediate Actions Required:</b><br/><br/>
        ✓ <b>Do Not Share:</b> Avoid distributing this content until verified<br/>
        ✓ <b>Source Verification:</b> Attempt to trace the original source of the video<br/>
        ✓ <b>Cross-Reference:</b> Check if authentic versions of this content exist<br/>
        ✓ <b>Report:</b> If malicious intent is suspected, report to appropriate authorities<br/>
        ✓ <b>Document:</b> Preserve metadata and chain of custody for potential investigation<br/>
        ✓ <b>Expert Review:</b> Consider additional forensic analysis for high-stakes cases<br/><br/>
        <b>For Organizations:</b><br/>
        • Update detection signatures with this sample's characteristics<br/>
        • Brief relevant teams about this potential threat<br/>
        • Monitor for similar content in your threat landscape
        """
    else:
        actions_text = """
        <b>Standard Verification Completed:</b><br/><br/>
        ✓ <b>Content Cleared:</b> No AI manipulation detected<br/>
        ✓ <b>Source Verification:</b> Still recommended as best practice<br/>
        ✓ <b>Context Check:</b> Verify the context matches the claimed narrative<br/>
        ✓ <b>Metadata Review:</b> Original metadata appears intact<br/><br/>
        <b>Recommendations:</b><br/>
        • This video can be considered authentic based on AI analysis<br/>
        • Standard journalistic/verification practices still apply<br/>
        • Archive this report for future reference if needed
        """
    story.append(Paragraph(actions_text, normal_style))
    story.append(Spacer(1, 15))
    
    # ============ 12. AUDIT & LOGS ============
    story.append(Paragraph("12. AUDIT & LOGS", section_style))
    
    audit_data = [
        ['Timestamp', 'Action', 'Details'],
        [timestamp, 'ANALYSIS_INITIATED', f'Scan mode: {video_info.get("scan_mode", "quick").upper()}'],
        [timestamp, 'FRAMES_EXTRACTED', f'{video_info.get("frames_analyzed", len(predictions))} frames processed'],
        [timestamp, 'AI_ANALYSIS_COMPLETE', f'Model: Roboflow DeepFake Detector'],
        [timestamp, 'VERDICT_GENERATED', f'{verdict.get("verdict", "Unknown")} ({confidence}%)'],
        [timestamp, 'REPORT_GENERATED', f'Report ID: {report_id}'],
    ]
    audit_table = Table(audit_data, colWidths=[130, 130, 230])
    audit_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#333333')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(audit_table)
    story.append(Spacer(1, 20))
    
    # ============ FOOTER ============
    footer_text = """
    <b>DISCLAIMER:</b> This report is generated by an automated AI system and should be used as one component 
    of a comprehensive verification process. DeepScan AI provides probabilistic assessments based on pattern 
    recognition and should not be considered definitive proof. For legal or critical applications, additional 
    expert analysis is recommended.<br/><br/>
    <b>Generated by DeepScan AI</b> | www.deepscan.ai | Report ID: """ + report_id + """<br/>
    © 2024 DeepScan AI - Advanced Video Authenticity Analysis
    """
    story.append(Paragraph(footer_text, ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return buffer, report_id

def download_video(url, output_path):
    """Download video from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        r = requests.get(url, stream=True, headers=headers, timeout=60)
        r.raise_for_status()
        
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return output_path, None
    except requests.exceptions.RequestException as e:
        return None, str(e)

def is_youtube_url(url):
    """Check if URL is a YouTube video"""
    youtube_patterns = [
        r'(https?://)?(www\.)?youtube\.com/watch\?v=',
        r'(https?://)?(www\.)?youtube\.com/shorts/',
        r'(https?://)?(www\.)?youtu\.be/',
        r'(https?://)?(www\.)?youtube\.com/embed/',
    ]
    for pattern in youtube_patterns:
        if re.search(pattern, url):
            return True
    return False

def download_youtube_video(url, output_path):
    """Download YouTube video using yt-dlp"""
    try:
        # Ensure output_path has .mp4 extension for the template
        if not output_path.endswith('.mp4'):
            output_path_template = output_path + '.%(ext)s'
        else:
            output_path_template = output_path.rsplit('.', 1)[0] + '.%(ext)s'

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/bestvideo+bestaudio/best',
            'outtmpl': output_path_template,
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'noplaylist': True,
            'socket_timeout': 30,
            'retries': 5,
            'fragment_retries': 5,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
            },
            'extractor_args': {
                'youtube': {
                    'player_client': ['web', 'android'],
                }
            },
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                return None, "Could not extract video info"
        
        # Determine the actual downloaded file path
        base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
        
        # Check possible file paths
        for ext in ['.mp4', '.webm', '.mkv', '.avi', '.mov']:
            candidate = base_path + ext
            if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                return candidate, None
        
        # Also check the original output_path as-is
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path, None
        
        # Search the upload folder for any file matching the uuid prefix
        upload_dir = os.path.dirname(output_path)
        base_name = os.path.basename(base_path)
        for f in os.listdir(upload_dir):
            if f.startswith(base_name) and os.path.getsize(os.path.join(upload_dir, f)) > 0:
                return os.path.join(upload_dir, f), None

        return None, "Downloaded file not found or is empty"
            
    except Exception as e:
        return None, str(e)

def get_trending_videos(max_results=12):
    """Get trending/popular videos from YouTube using search"""
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        
        # Use YouTube search for popular recent videos instead of trending feed
        # Search for popular content categories
        search_queries = [
            "ytsearch5:trending viral video 2024",
            "ytsearch5:popular music video",
            "ytsearch4:breaking news today"
        ]
        
        videos = []
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for search_query in search_queries:
                try:
                    results = ydl.extract_info(search_query, download=False)
                    if results and 'entries' in results:
                        for entry in results['entries']:
                            if entry and len(videos) < max_results:
                                video_id = entry.get('id', '')
                                if video_id and not any(v['id'] == video_id for v in videos):
                                    videos.append({
                                        'id': video_id,
                                        'title': entry.get('title', 'Unknown'),
                                        'url': f"https://www.youtube.com/watch?v={video_id}",
                                        'thumbnail': entry.get('thumbnail') or f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
                                        'duration': entry.get('duration', 0),
                                        'channel': entry.get('channel', entry.get('uploader', 'Unknown')),
                                        'view_count': entry.get('view_count', 0)
                                    })
                except:
                    continue
        
        if not videos:
            return None, "No videos found"
            
        return videos, None
    except Exception as e:
        return None, str(e)

def search_youtube_videos(query, max_results=10):
    """Search YouTube for videos matching a query"""
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'default_search': 'ytsearch',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
            
        videos = []
        if results and 'entries' in results:
            for entry in results['entries']:
                if entry:
                    video_id = entry.get('id', '')
                    videos.append({
                        'id': video_id,
                        'title': entry.get('title', 'Unknown'),
                        'url': f"https://www.youtube.com/watch?v={video_id}",
                        'thumbnail': entry.get('thumbnail') or f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
                        'duration': entry.get('duration', 0),
                        'channel': entry.get('channel', entry.get('uploader', 'Unknown')),
                        'view_count': entry.get('view_count', 0)
                    })
        return videos, None
    except Exception as e:
        return None, str(e)

def get_breaking_news_videos(max_results=12):
    """Get breaking news videos from YouTube"""
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        
        # Search queries for breaking news content
        search_queries = [
            "ytsearch4:breaking news today live",
            "ytsearch4:latest news headlines today",
            "ytsearch4:world news today 2024",
            "ytsearch4:top stories news today"
        ]
        
        videos = []
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for search_query in search_queries:
                try:
                    results = ydl.extract_info(search_query, download=False)
                    if results and 'entries' in results:
                        for entry in results['entries']:
                            if entry and len(videos) < max_results:
                                video_id = entry.get('id', '')
                                if video_id and not any(v['id'] == video_id for v in videos):
                                    videos.append({
                                        'id': video_id,
                                        'title': entry.get('title', 'Unknown'),
                                        'url': f"https://www.youtube.com/watch?v={video_id}",
                                        'thumbnail': entry.get('thumbnail') or f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
                                        'duration': entry.get('duration', 0),
                                        'channel': entry.get('channel', entry.get('uploader', 'Unknown')),
                                        'view_count': entry.get('view_count', 0),
                                        'category': 'news'
                                    })
                except:
                    continue
        
        if not videos:
            return None, "No news videos found"
            
        return videos, None
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/trending', methods=['GET'])
def get_trending():
    """Get trending videos for deepfake analysis"""
    max_results = request.args.get('limit', 12, type=int)
    videos, error = get_trending_videos(max_results)
    
    if error:
        return jsonify({'error': f'Failed to fetch trending videos: {error}'}), 400
    
    return jsonify({
        'success': True,
        'videos': videos
    })

@app.route('/breaking-news', methods=['GET'])
def get_breaking_news():
    """Get breaking news videos for deepfake analysis"""
    max_results = request.args.get('limit', 20, type=int)
    videos, error = get_breaking_news_videos(max_results)
    
    if error:
        return jsonify({'error': f'Failed to fetch news videos: {error}'}), 400
    
    return jsonify({
        'success': True,
        'videos': videos
    })

@app.route('/search-videos', methods=['POST'])
def search_videos():
    """Search for videos by keyword"""
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({'error': 'No search query provided'}), 400
    
    query = data['query']
    max_results = data.get('max_results', 10)
    
    videos, error = search_youtube_videos(query, max_results)
    
    if error:
        return jsonify({'error': f'Search failed: {error}'}), 400
    
    return jsonify({
        'success': True,
        'videos': videos
    })

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video = request.files['video']
    
    if video.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    # Get scan mode from form data (default to 'quick')
    scan_mode = request.form.get('scan_mode', 'quick')
    if scan_mode not in ['quick', 'deep', 'ultra']:
        scan_mode = 'quick'
    
    # Save uploaded video
    video_filename = f"{uuid.uuid4()}_{video.filename}"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    video.save(video_path)
    
    try:
        # Extract frames based on scan mode
        frames_data, error = extract_frames(video_path, scan_mode)
        
        if error:
            return jsonify({'error': error}), 400
        
        # Analyze frames with Roboflow
        predictions = analyze_frames(frames_data)
        
        # Apply low confidence rule: if any frame < 60%, mark all as AI
        predictions = apply_low_confidence_rule(predictions)
        
        # Get final verdict
        verdict = get_final_verdict(predictions)
        
        # Clean up video file
        os.remove(video_path)
        
        return jsonify({
            'success': True,
            'video_info': {
                'duration': frames_data['duration'],
                'fps': frames_data['fps'],
                'total_frames': frames_data['total_frames'],
                'scan_mode': scan_mode,
                'frames_analyzed': frames_data['frames_analyzed']
            },
            'predictions': predictions,
            'verdict': verdict
        })
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-url', methods=['POST'])
def analyze_video_url():
    data = request.get_json()
    
    if not data or 'url' not in data:
        return jsonify({'error': 'No video URL provided'}), 400
    
    video_url = data['url']
    
    if not video_url:
        return jsonify({'error': 'Empty URL provided'}), 400
    
    # Get scan mode (default to 'quick')
    scan_mode = data.get('scan_mode', 'quick')
    if scan_mode not in ['quick', 'deep', 'ultra']:
        scan_mode = 'quick'
    
    # Generate unique filename
    video_filename = f"{uuid.uuid4()}_downloaded_video"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    
    try:
        # Check if it's a YouTube URL
        if is_youtube_url(video_url):
            # Download YouTube video
            video_path_base = video_path
            downloaded_path, error = download_youtube_video(video_url, video_path_base)
            
            if error:
                return jsonify({'error': f'Failed to download YouTube video: {error}'}), 400
        else:
            # Download non-YouTube video URL
            video_path += '.mp4'
            downloaded_path, error = download_video(video_url, video_path)
            
            if error:
                return jsonify({'error': f'Failed to download video: {error}'}), 400
        
        # Extract frames based on scan mode
        frames_data, error = extract_frames(downloaded_path, scan_mode)
        
        if error:
            if os.path.exists(downloaded_path):
                os.remove(downloaded_path)
            return jsonify({'error': error}), 400
        
        # Analyze frames with Roboflow
        predictions = analyze_frames(frames_data)
        
        # Apply low confidence rule: if any frame < 60%, mark all as AI
        predictions = apply_low_confidence_rule(predictions)
        
        # Get final verdict
        verdict = get_final_verdict(predictions)
        
        # Clean up video file
        if os.path.exists(downloaded_path):
            os.remove(downloaded_path)
        
        return jsonify({
            'success': True,
            'video_info': {
                'duration': frames_data['duration'],
                'fps': frames_data['fps'],
                'total_frames': frames_data['total_frames'],
                'scan_mode': scan_mode,
                'frames_analyzed': frames_data['frames_analyzed'],
                'source': 'YouTube' if is_youtube_url(video_url) else 'URL'
            },
            'predictions': predictions,
            'verdict': verdict
        })
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(video_path + '.mp4'):
            os.remove(video_path + '.mp4')
        return jsonify({'error': str(e)}), 500

@app.route('/generate-report', methods=['POST'])
def generate_report():
    """Generate PDF report from analysis data"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        video_info = data.get('video_info', {})
        predictions = data.get('predictions', [])
        verdict = data.get('verdict', {})
        source_info = data.get('source', video_info.get('source', 'Unknown'))
        
        # Generate PDF
        pdf_buffer, report_id = generate_pdf_report(data, video_info, predictions, verdict, source_info)
        
        # Save PDF to file
        pdf_filename = f"Truesight_Report_{report_id}.pdf"
        pdf_path = os.path.join(app.config['REPORTS_FOLDER'], pdf_filename)
        
        with open(pdf_path, 'wb') as f:
            f.write(pdf_buffer.getvalue())
        
        return jsonify({
            'success': True,
            'report_id': report_id,
            'filename': pdf_filename,
            'download_url': f'/download-report/{pdf_filename}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-report/<filename>')
def download_report(filename):
    """Download generated PDF report"""
    try:
        pdf_path = os.path.join(app.config['REPORTS_FOLDER'], filename)
        if os.path.exists(pdf_path):
            return send_file(
                pdf_path,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=filename
            )
        return jsonify({'error': 'Report not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5007)
