const { plan, simulate } = require("./planner_page.js");
const cases = JSON.parse(require("fs").readFileSync("cases.json", "utf8"));
const out = cases.map(c => { const P = { vc: c.P.vc, start: c.P.start, rel: c.P.rel, top: c.P.top, flow: c.P.flow, eff: c.P.eff, vl: c.P.vl, rate: c.P.rate, plan: c.P.plan, first: c.P.first, silentOk: c.P.silent_ok, place: c.P.place };
  const w = plan(c.text, P); const sim = simulate(w, P, null);
  return { br: w.map(o => o.br || null), dv: w.map(o => o.dv || 0), bdur: w.map(o => o.bdur || 0), end: sim.end, minV: sim.minV }; });
require("fs").writeFileSync("js_out.json", JSON.stringify(out));
