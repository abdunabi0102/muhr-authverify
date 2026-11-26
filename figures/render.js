const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [dir, outdir] = process.argv.slice(2);
  const b = await chromium.launch();
  for (const name of ['fig1_framework', 'fig3_network']) {
    const svg = fs.readFileSync(`${dir}/${name}.svg`, 'utf8');
    const m = svg.match(/width="(\d+)" height="(\d+)"/); const w = +m[1], h = +m[2];
    const page = await b.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 2 });
    await page.setContent(`<html><body style="margin:0">${svg}</body></html>`);
    await page.waitForTimeout(300);
    await page.screenshot({ path: `${outdir}/${name}.png`, clip: { x: 0, y: 0, width: w, height: h } });
    await page.pdf({ path: `${outdir}/${name}.pdf`, width: `${w}px`, height: `${h}px`, printBackground: true, pageRanges: '1' });
    await page.close();
  }
  await b.close(); console.log('rendered');
})();
