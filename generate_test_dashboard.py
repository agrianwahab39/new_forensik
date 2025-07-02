# --- START OF FILE generate_test_dashboard.py ---

import xml.etree.ElementTree as ET
from datetime import datetime
import os

def get_color_for_coverage(percentage):
    """Mengembalikan warna berdasarkan persentase cakupan."""
    if percentage >= 90:
        return "#4CAF50"  # Hijau
    elif percentage >= 70:
        return "#FFC107"  # Kuning
    else:
        return "#F44336"  # Merah

def generate_dashboard(xml_file="coverage.xml", output_file="test_dashboard.html"):
    """Membaca file coverage.xml dan menghasilkan dashboard HTML."""
    
    if not os.path.exists(xml_file):
        print(f"Error: File '{xml_file}' tidak ditemukan. Jalankan 'coverage xml' terlebih dahulu.")
        return

    # Parse file XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Ekstrak metrik global
    line_rate = float(root.get("line-rate", 0)) * 100
    branch_rate = float(root.get("branch-rate", 0)) * 100
    
    # Kumpulkan data per file
    files_data = []
    packages = root.find("packages")
    for package in packages:
        classes = package.find("classes")
        for cls in classes:
            filename = cls.get("filename")
            file_line_rate = float(cls.get("line-rate", 0)) * 100
            file_branch_rate = float(cls.get("branch-rate", 0)) * 100
            
            # Ekstrak jumlah statement dan branch
            lines = cls.find("lines")
            total_statements = len(lines)
            missed_statements = sum(1 for line in lines if line.get('hits') == '0')
            
            # Hitung branch (jika ada)
            branches = cls.find("branches")
            total_branches = int(branches.get('total', 0)) if branches is not None else 0
            missed_branches = int(branches.get('covered', 0)) if branches is not None else 0
            missed_branches = total_branches - missed_branches # Koreksi: `covered` adalah yg ter-cover
            
            files_data.append({
                "name": filename,
                "line_coverage": file_line_rate,
                "branch_coverage": file_branch_rate,
                "total_statements": total_statements,
                "missed_statements": missed_statements,
                "total_branches": total_branches,
                "missed_branches": missed_branches,
            })

    # Urutkan berdasarkan cakupan terendah untuk menyorot masalah
    files_data.sort(key=lambda x: x["line_coverage"])

    # Buat konten HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <title>Dashboard Laporan Pengujian & Coverage</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 40px; background-color: #f4f7f9; color: #333; }}
            .container {{ max-width: 1200px; margin: auto; }}
            .header {{ text-align: center; margin-bottom: 40px; }}
            .summary-cards {{ display: flex; gap: 20px; justify-content: center; text-align: center; }}
            .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); flex-grow: 1; }}
            .card h3 {{ margin-top: 0; color: #555; }}
            .coverage-number {{ font-size: 3em; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 40px; background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f8f8f8; }}
            .progress-bar-bg {{ background-color: #e9ecef; border-radius: 5px; height: 20px; }}
            .progress-bar {{ height: 100%; border-radius: 5px; color: white; text-align: center; line-height: 20px; font-size: 12px; }}
            footer {{ text-align: center; margin-top: 40px; color: #888; }}
            a {{ color: #007bff; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Dashboard Laporan Pengujian & Coverage</h1>
                <p>Laporan Dihasilkan pada: {datetime.now().strftime('%d %B %Y, %H:%M:%S')}</p>
            </div>

            <div class="summary-cards">
                <div class="card">
                    <h3>Cakupan Baris (Line)</h3>
                    <div class="coverage-number" style="color: {get_color_for_coverage(line_rate)};">{line_rate:.1f}%</div>
                </div>
                <div class="card">
                    <h3>Cakupan Cabang (Branch)</h3>
                    <div class="coverage-number" style="color: {get_color_for_coverage(branch_rate)};">{branch_rate:.1f}%</div>
                </div>
                <div class="card">
                    <h3>Tautan Detail</h3>
                    <p><a href="./coverage_html_report/index.html" target="_blank">Buka Laporan HTML Rinci</a></p>
                    <p>Laporan ini memberikan detail per baris kode.</p>
                </div>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>Nama File</th>
                        <th>Cakupan Baris</th>
                        <th>Cakupan Cabang</th>
                        <th>Pernyataan (Stmts)</th>
                        <th>Cabang (Branch)</th>
                    </tr>
                </thead>
                <tbody>
    """

    # Tambahkan baris untuk setiap file
    for file in files_data:
        line_color = get_color_for_coverage(file["line_coverage"])
        branch_color = get_color_for_coverage(file["branch_coverage"])
        
        html_content += f"""
                    <tr>
                        <td>{file["name"]}</td>
                        <td>
                            <div class="progress-bar-bg">
                                <div class="progress-bar" style="width: {file['line_coverage']:.1f}%; background-color: {line_color};">
                                    {file['line_coverage']:.1f}%
                                </div>
                            </div>
                        </td>
                        <td>
                            <div class="progress-bar-bg">
                                <div class="progress-bar" style="width: {file['branch_coverage']:.1f}%; background-color: {branch_color};">
                                    {file['branch_coverage']:.1f}%
                                </div>
                            </div>
                        </td>
                        <td>{file["total_statements"]} ({file["missed_statements"]} terlewat)</td>
                        <td>{file["total_branches"]} ({file["missed_branches"]} terlewat)</td>
                    </tr>
        """
        
    html_content += """
                </tbody>
            </table>
            
            <footer>
                <p>Dashboard ini dihasilkan secara otomatis dari data pengujian dan cakupan kode.</p>
            </footer>
        </div>
    </body>
    </html>
    """

    # Tulis ke file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ Dashboard berhasil dibuat: {output_file}")


if __name__ == "__main__":
    generate_dashboard()