# app/routes/health.py
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
from flask import Blueprint, jsonify, request, send_file
from app.extensions import service, db, logger
from datetime import datetime
import io
analytics_bp = Blueprint('analytic', __name__)

@analytics_bp.route('/health', methods=['GET'])
def health_check():
    status = {'status': 'healthy', 'components': {}}
    # Check DB
    try:
        service.db.fetchone("SELECT 1")
        status['components']['database'] = 'healthy'
    except:
        status['components']['database'] = 'unhealthy'
        status['status'] = 'degraded'
    # Check Redis
    try:
        service.redis.ping()
        status['components']['redis'] = 'healthy'
    except:
        status['components']['redis'] = 'unhealthy'
    
    return jsonify(status)

@analytics_bp.route('/api/metrics', methods=['GET'])
def get_metrics():
    try:
        days = request.args.get('days', 7, type=int)
        metrics = service.get_metrics(days)
        return jsonify({'success': True, 'metrics': metrics})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
# ==============================================================================
# 1. API DASHBOARD DATA (Cho Frontend vẽ biểu đồ)
# ==============================================================================
@analytics_bp.route('/api/analytics/dashboard', methods=['GET'])
def get_dashboard_stats():
    """
    Get detailed metrics for admin dashboard
    Query Params: days (default 30)
    """
    try:
        days = request.args.get('days', 30, type=int)
        
        # A. Thống kê tổng quan (Summary)
        summary_query = """
            SELECT 
                SUM(impressions) as total_impressions,
                SUM(clicks) as total_clicks,
                SUM(conversions) as total_orders,
                SUM(revenue) as total_revenue,
                CASE WHEN SUM(impressions) > 0 THEN 
                     CAST(SUM(clicks) AS DECIMAL) / SUM(impressions) * 100 
                ELSE 0 END as avg_ctr,
                CASE WHEN SUM(clicks) > 0 THEN 
                     CAST(SUM(conversions) AS DECIMAL) / SUM(clicks) * 100 
                ELSE 0 END as avg_cv_rate
            FROM daily_recommendation_stats
            WHERE date >= CURRENT_DATE - INTERVAL '%s days'
        """
        summary = db.fetchone(summary_query, (days,))
        
        # B. Biểu đồ theo thời gian (Line Chart Data)
        trend_query = """
            SELECT 
                TO_CHAR(date, 'YYYY-MM-DD') as date,
                SUM(impressions) as impressions,
                SUM(clicks) as clicks,
                SUM(conversions) as orders,
                SUM(revenue) as revenue
            FROM daily_recommendation_stats
            WHERE date >= CURRENT_DATE - INTERVAL '%s days'
            GROUP BY date
            ORDER BY date ASC
        """
        trend_df = db.query(trend_query, (days,))
        
        # C. Hiệu quả theo thuật toán (Pie Chart Data)
        algo_query = """
            SELECT 
                rec_type,
                SUM(clicks) as clicks,
                SUM(revenue) as revenue
            FROM daily_recommendation_stats
            WHERE date >= CURRENT_DATE - INTERVAL '%s days'
            GROUP BY rec_type
        """
        algo_df = db.query(algo_query, (days,))
        
        return jsonify({
            'success': True,
            'period': f'Last {days} days',
            'summary': {
                'total_impressions': int(summary[0] or 0),
                'total_clicks': int(summary[1] or 0),
                'total_orders': int(summary[2] or 0),
                'total_revenue': float(summary[3] or 0),
                'ctr': round(float(summary[4] or 0), 2),
                'conversion_rate': round(float(summary[5] or 0), 2)
            },
            'trend_chart': trend_df.to_dict('records'),
            'algorithm_performance': algo_df.to_dict('records')
        })

    except Exception as e:
        logger.error(f"Dashboard API Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==============================================================================
# 2. API EXPORT REPORT (PDF/CSV)
# ==============================================================================
@analytics_bp.route('/api/analytics/export', methods=['GET'])
def export_report():
    """
    Export report file
    Query Params: 
      - days: int (default 30)
      - format: 'csv' or 'pdf' (default 'csv')
    """
    try:
        days = request.args.get('days', 30, type=int)
        fmt = request.args.get('format', 'csv').lower()
        
        # Lấy dữ liệu chi tiết
        query = """
            SELECT 
                date,
                rec_type,
                impressions,
                clicks,
                conversions,
                ctr,
                conversion_rate,
                revenue,
                avg_order_value
            FROM daily_recommendation_stats
            WHERE date >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY date DESC, rec_type
        """
        df = db.query(query, (days,))
        
        if df.empty:
            return jsonify({'error': 'No data found'}), 404

        # --- XUẤT CSV ---
        if fmt == 'csv':
            output = io.BytesIO()
            df.to_csv(output, index=False, encoding='utf-8')
            output.seek(0)
            
            return send_file(
                output,
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'recommendation_report_{datetime.now().strftime("%Y%m%d")}.csv'
            )
            
        # --- XUẤT PDF ---
        elif fmt == 'pdf':
            output = io.BytesIO()
            doc = SimpleDocTemplate(output, pagesize=letter)
            elements = []
            styles = getSampleStyleSheet()
            
            # Title
            elements.append(Paragraph(f"Recommendation Performance Report", styles['Title']))
            elements.append(Paragraph(f"Period: Last {days} days - Generated: {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
            elements.append(Spacer(1, 12))
            
            # Convert DF to list of lists for Table
            data = [df.columns.tolist()] + df.values.tolist()
            
            # Create Table
            t = Table(data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(t)
            doc.build(elements)
            
            output.seek(0)
            return send_file(
                output,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'recommendation_report_{datetime.now().strftime("%Y%m%d")}.pdf'
            )
        
        else:
            return jsonify({'error': 'Invalid format. Use csv or pdf'}), 400

    except Exception as e:
        logger.error(f"Export Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500