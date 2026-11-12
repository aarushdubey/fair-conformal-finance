import os
import base64
import subprocess
import fitz

def main():
    print("Building FairTransCP Project Report...")
    
    logo_path = "/Users/aarushdubey/Downloads/AISTAT2027/AISTAT2027/report_assets/page1_img1_26.png"
    pareto_path = "/Users/aarushdubey/Downloads/AISTAT2027/AISTAT2027/fair-conformal-finance/paper/figures/pareto_coverage_vs_disparity.png"
    disparity_path = "/Users/aarushdubey/Downloads/AISTAT2027/AISTAT2027/fair-conformal-finance/paper/figures/set_size_disparity_comparison.png"
    template_path = "/Users/aarushdubey/Downloads/AISTAT2027/AISTAT2027/report_template.html"
    
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode("utf-8")
    with open(pareto_path, "rb") as f:
        pareto_b64 = base64.b64encode(f.read()).decode("utf-8")
    with open(disparity_path, "rb") as f:
        disparity_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
        
    html = html.replace("###LOGO_B64###", logo_b64)
    html = html.replace("###PARETO_B64###", pareto_b64)
    html = html.replace("###DISPARITY_B64###", disparity_b64)
    
    rendered_html_path = "/Users/aarushdubey/Downloads/AISTAT2027/AISTAT2027/report_rendered.html"
    with open(rendered_html_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    raw_pdf = "/Users/aarushdubey/Downloads/AISTAT2027/AISTAT2027/report_raw.pdf"
    print("Invoking Chrome headless to generate PDF...")
    chrome_cmd = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={raw_pdf}",
        rendered_html_path
    ]
    subprocess.run(chrome_cmd, check=True)
    print("Chrome rendering complete.")
    
    # Post-process with PyMuPDF
    print("Post-processing PDF...")
    doc = fitz.open(raw_pdf)
    total_pages = len(doc)
    print(f"Total pages: {total_pages}")
    
    final_pdf_1 = "/Users/aarushdubey/Downloads/AISTAT2027/AISTAT2027/FairTransCP_Project_Report.pdf"
    final_pdf_2 = "/Users/aarushdubey/Downloads/FairTransCP_Project_Report.pdf"
    
    # Map pages to their displayed page numbers in Index
    # Page 1: Cover (no number)
    # Page 2: Abstract (no number)
    # Page 3: Index (no number)
    # Pages 4 to end: Numbered 01, 02, ...
    for page_idx in range(total_pages):
        page = doc[page_idx]
        if page_idx >= 3:
            logical_page_num = page_idx - 2
            page_str = f"{logical_page_num:02d}"
            
            # Bottom right page number
            page.insert_text(fitz.Point(515, 805), page_str, fontname="times-roman", fontsize=11, color=(0.2, 0.2, 0.2))
            
            # Running top header
            header_text = "FairTransCP: Balancing Coverage and Set-Size Equity in Credit Scoring"
            page.insert_text(fitz.Point(55, 45), header_text, fontname="times-italic", fontsize=9, color=(0.4, 0.4, 0.4))
            page.draw_line(fitz.Point(55, 50), fitz.Point(540, 50), color=(0.7, 0.7, 0.7), width=0.5)

    doc.save(final_pdf_1)
    doc.save(final_pdf_2)
    print("Saved final reports to:")
    print(f"  -> {final_pdf_1}")
    print(f"  -> {final_pdf_2}")

if __name__ == "__main__":
    main()

