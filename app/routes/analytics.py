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
from datetime import datetime, timedelta
import calendar # Để lấy ngày cuối cùng của tháng
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
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta # Cần cài: pip install python-dateutil

# Hàm hỗ trợ xử lý ngày tháng (Helper Function)
def get_date_range(args):
    """
    Xử lý các tham số query: days, month, year, start_date/end_date
    Trả về: (start_date_str, end_date_str)
    """
    now = datetime.now()
    end_date = now

    # 1. Lọc theo khoảng ngày cụ thể (start_date & end_date)
    if args.get('start_date') and args.get('end_date'):
        return args.get('start_date'), args.get('end_date')

    # 2. Lọc theo tháng (month=2023-10)
    if args.get('month'):
        try:
            date_obj = datetime.strptime(args.get('month'), '%Y-%m')
            start_date = date_obj.replace(day=1)
            # Ngày cuối tháng = ngày đầu tháng + 1 tháng - 1 ngày
            end_date = start_date + relativedelta(months=1) - timedelta(days=1)
            return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        except:
            pass # Fallback về default

    # 3. Lọc theo năm (year=2024)
    if args.get('year'):
        try:
            year = int(args.get('year'))
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
            return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        except:
            pass

    # 4. Mặc định: Lọc theo số ngày lùi lại (days=7)
    days = args.get('days', 30, type=int)
    start_date = now - timedelta(days=days)
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

@analytics_bp.route('/api/analytics/shop/<shop_id>', methods=['GET'])
def get_shop_stats(shop_id):
    try:
        # Lấy khoảng thời gian từ helper
        start_date, end_date = get_date_range(request.args)
        
        # --- A. Thống kê tổng quan (Summary) ---
        # SỬA LỖI: Thêm dấu phẩy sau search_clicks
        query_stats = """
            SELECT 
                COUNT(CASE WHEN action_type = 'view' THEN 1 END) as views,               -- Index 0
                COUNT(CASE WHEN action_type = 'click' THEN 1 END) as total_clicks,       -- Index 1
                COUNT(CASE WHEN action_type = 'cart_add' THEN 1 END) as add_to_carts,    -- Index 2
                COUNT(CASE WHEN action_type = 'purchase' THEN 1 END) as orders,          -- Index 3
                COUNT(CASE WHEN action_type = 'click' AND metadata->>'source' = 'search' THEN 1 END) as search_clicks, -- Index 4
                COALESCE(SUM(CASE WHEN action_type = 'purchase' THEN price * quantity ELSE 0 END), 0) as revenue      -- Index 5
            FROM user_interactions
            WHERE shop_id = %s
            AND created_at BETWEEN %s AND %s
        """
        stats = service.db.fetchone(query_stats, (shop_id, start_date, end_date))
        
        # Xử lý trường hợp shop mới chưa có data (stats là None)
        if not stats:
            stats = (0, 0, 0, 0, 0, 0)

        # SỬA LỖI: Mapping đúng index
        views = int(stats[0] or 0)
        clicks = int(stats[1] or 0)
        carts = int(stats[2] or 0)
        orders = int(stats[3] or 0)
        search_clicks = int(stats[4] or 0)
        revenue = float(stats[5] or 0)

        # Tính tỷ lệ chuyển đổi (View -> Order)
        cv_rate = round((orders / views * 100), 2) if views > 0 else 0

        # --- B. Top sản phẩm ---
        query_top_products = """
            SELECT 
                product_id,
                COUNT(CASE WHEN action_type = 'view' THEN 1 END) as views,
                COUNT(CASE WHEN action_type = 'purchase' THEN 1 END) as orders,
                COALESCE(SUM(CASE WHEN action_type = 'purchase' THEN price * quantity ELSE 0 END), 0) as revenue
            FROM user_interactions
            WHERE shop_id = %s
            AND created_at BETWEEN %s AND %s
            GROUP BY product_id
            ORDER BY views DESC
            LIMIT 10
        """
        top_products_df = service.db.query(query_top_products, (shop_id, start_date, end_date))

        # --- C. Biểu đồ xu hướng (Trend Chart) ---
        query_trend = """
            SELECT 
                DATE(created_at) as date,
                COUNT(CASE WHEN action_type = 'view' THEN 1 END) as views,
                COUNT(CASE WHEN action_type = 'purchase' THEN 1 END) as orders,
                COALESCE(SUM(CASE WHEN action_type = 'purchase' THEN price * quantity ELSE 0 END), 0) as revenue
            FROM user_interactions
            WHERE shop_id = %s
            AND created_at BETWEEN %s AND %s
            GROUP BY DATE(created_at)
            ORDER BY date ASC
        """
        trend_df = service.db.query(query_trend, (shop_id, start_date, end_date))

        if not trend_df.empty:
            trend_df['date'] = trend_df['date'].astype(str)

        return jsonify({
            'success': True,
            'shop_id': shop_id,
            'period': {
                'start': str(start_date),
                'end': str(end_date)
            },
            'summary': {
                'views': views,
                'total_clicks': clicks,
                'add_to_carts': carts,
                'orders': orders,
                'search_clicks': search_clicks,
                'revenue': revenue,
                'conversion_rate': cv_rate
            },
            'trend_chart': trend_df.to_dict('records'),
            'top_products': top_products_df.to_dict('records')
        })

    except Exception as e:
        logger.error(f"Shop Analytics Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==============================================================================
# 4. API THỐNG KÊ SẢN PHẨM (PRODUCT ANALYTICS)
# ==============================================================================
# app/routes/analytics.py

@analytics_bp.route('/api/analytics/product/<product_id>', methods=['GET'])
def get_product_stats_detailed(product_id):
    """
    Thống kê chi tiết Product hỗ trợ lọc:
    - ?days=7
    - ?start_date=...&end_date=...
    - ?month=...
    - ?year=...
    """
    try:
        # 1. Lấy khoảng thời gian (Dùng lại hàm helper đã khai báo ở trên)
        start_date, end_date = get_date_range(request.args)
        
        # 2. Lấy thông tin cơ bản của sản phẩm (Giá hiện tại, Rating, Category)
        # Để Frontend hiển thị tên/giá bên cạnh biểu đồ
        query_info = """
            SELECT category_id, current_price, avg_rating_updated, view_count_30d 
            FROM product_features 
            WHERE product_id = %s
        """
        prod_info = service.db.fetchone(query_info, (product_id,))
        
        # --- A. Thống kê tổng quan (Summary) ---
        query_stats = """
            SELECT 
                COUNT(CASE WHEN action_type = 'view' THEN 1 END) as views,               -- 0
                COUNT(CASE WHEN action_type = 'click' THEN 1 END) as total_clicks,       -- 1
                COUNT(CASE WHEN action_type = 'cart_add' THEN 1 END) as add_to_carts,    -- 2
                COUNT(CASE WHEN action_type = 'purchase' THEN 1 END) as orders,          -- 3
                COUNT(CASE WHEN action_type = 'click' AND metadata->>'source' = 'search' THEN 1 END) as search_clicks, -- 4
                COALESCE(SUM(CASE WHEN action_type = 'purchase' THEN price * quantity ELSE 0 END), 0) as revenue      -- 5
            FROM user_interactions
            WHERE product_id = %s
            AND created_at BETWEEN %s AND %s
        """
        stats = service.db.fetchone(query_stats, (product_id, start_date, end_date))
        
        if not stats:
            stats = (0, 0, 0, 0, 0, 0)

        views = int(stats[0] or 0)
        clicks = int(stats[1] or 0)
        carts = int(stats[2] or 0)
        orders = int(stats[3] or 0)
        search_clicks = int(stats[4] or 0)
        revenue = float(stats[5] or 0)

        # Tính tỷ lệ chuyển đổi
        ctr = round((clicks / views * 100), 2) if views > 0 else 0
        cv_rate = round((orders / views * 100), 2) if views > 0 else 0
        cart_rate = round((carts / views * 100), 2) if views > 0 else 0

        # --- B. Biểu đồ xu hướng (Trend Chart) ---
        query_trend = """
            SELECT 
                DATE(created_at) as date,
                COUNT(CASE WHEN action_type = 'view' THEN 1 END) as views,
                COUNT(CASE WHEN action_type = 'cart_add' THEN 1 END) as carts,
                COUNT(CASE WHEN action_type = 'purchase' THEN 1 END) as orders,
                COALESCE(SUM(CASE WHEN action_type = 'purchase' THEN price * quantity ELSE 0 END), 0) as revenue
            FROM user_interactions
            WHERE product_id = %s
            AND created_at BETWEEN %s AND %s
            GROUP BY DATE(created_at)
            ORDER BY date ASC
        """
        trend_df = service.db.query(query_trend, (product_id, start_date, end_date))

        if not trend_df.empty:
            trend_df['date'] = trend_df['date'].astype(str)

        return jsonify({
            'success': True,
            'product_id': product_id,
            'info': {
                'category_id': prod_info[0] if prod_info else None,
                'current_price': float(prod_info[1]) if prod_info else 0,
                'rating': float(prod_info[2]) if prod_info and prod_info[2] else 0,
                'total_views_30d': int(prod_info[3]) if prod_info else 0
            },
            'period': {
                'start': str(start_date),
                'end': str(end_date)
            },
            'summary': {
                'views': views,
                'total_clicks': clicks,
                'add_to_carts': carts,
                'orders': orders,
                'search_clicks': search_clicks,
                'revenue': revenue,
                'ctr': ctr,                 # Tỷ lệ click xem
                'cart_rate': cart_rate,     # Tỷ lệ thêm giỏ
                'conversion_rate': cv_rate  # Tỷ lệ mua hàng
            },
            'trend_chart': trend_df.to_dict('records')
        })

    except Exception as e:
        logger.error(f"Product Analytics Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
# app/routes/analytics.py

@analytics_bp.route('/api/analytics/performance', methods=['GET'])
def get_recommendation_performance():
    """
    Thống kê hiệu quả hệ thống AI (A/B Testing Metric)
    Hỗ trợ lọc: ?days=30 hoặc ?start_date=...&end_date=...
    """
    try:
        # 1. Lấy khoảng thời gian (Dùng lại hàm helper get_date_range)
        start_date, end_date = get_date_range(request.args)

        # --- A. Thống kê theo từng Thuật toán (Algorithm Breakdown) ---
        # Group by rec_type (personalized, similar, cross-sell...)
        query_algo = """
            SELECT 
                rec_type,
                COUNT(*) as impressions,
                COUNT(clicked_at) as clicks,
                COUNT(purchased_at) as orders,
                COALESCE(SUM(purchase_amount), 0) as revenue
            FROM recommendation_logs
            WHERE shown_at BETWEEN %s AND %s
            GROUP BY rec_type
            ORDER BY revenue DESC
        """
        algo_rows = service.db.query(query_algo, (start_date, end_date))

        # Xử lý số liệu chi tiết cho từng thuật toán
        algo_stats = []
        total_summary = {
            'impressions': 0, 'clicks': 0, 'orders': 0, 'revenue': 0.0
        }

        for _, row in algo_rows.iterrows():
            impr = int(row['impressions'])
            clks = int(row['clicks'])
            ords = int(row['orders'])
            rev = float(row['revenue'])

            # Cộng dồn tổng
            total_summary['impressions'] += impr
            total_summary['clicks'] += clks
            total_summary['orders'] += ords
            total_summary['revenue'] += rev

            algo_stats.append({
                'algorithm': row['rec_type'],
                'impressions': impr,
                'clicks': clks,
                'orders': ords,
                'revenue': rev,
                'ctr': round((clks / impr * 100), 2) if impr > 0 else 0,
                'conversion_rate': round((ords / clks * 100), 2) if clks > 0 else 0
            })

        # Tính chỉ số tổng quan toàn hệ thống
        total_summary['ctr'] = round((total_summary['clicks'] / total_summary['impressions'] * 100), 2) if total_summary['impressions'] > 0 else 0
        total_summary['conversion_rate'] = round((total_summary['orders'] / total_summary['clicks'] * 100), 2) if total_summary['clicks'] > 0 else 0

        # --- B. Biểu đồ xu hướng (Trend Chart) ---
        # Để vẽ biểu đồ đường xem hiệu quả theo ngày
        query_trend = """
            SELECT 
                DATE(shown_at) as date,
                COUNT(*) as impressions,
                COUNT(clicked_at) as clicks,
                COALESCE(SUM(purchase_amount), 0) as revenue
            FROM recommendation_logs
            WHERE shown_at BETWEEN %s AND %s
            GROUP BY DATE(shown_at)
            ORDER BY date ASC
        """
        trend_df = service.db.query(query_trend, (start_date, end_date))
        
        if not trend_df.empty:
            trend_df['date'] = trend_df['date'].astype(str)

        return jsonify({
            'success': True,
            'period': {
                'start': str(start_date),
                'end': str(end_date)
            },
            'summary': total_summary,         # Tổng quan (Số to ở trên cùng dashboard)
            'by_algorithm': algo_stats,       # Bảng so sánh hoặc Pie Chart
            'trend_chart': trend_df.to_dict('records') # Line chart
        })

    except Exception as e:
        logger.error(f"Rec Performance Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500