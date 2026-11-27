// POST /api/team/checkout — a signed-in user creates a Gitsum team and gets a Stripe Checkout URL.
// Body: { name, seats (2–500), returnPath? } → { url, team_id }
// 401 login_required · 400 bad_request · 503 not_configured | capacity

import { env, fail, json, missingEnv, originOf, readJson } from "../../server/http.js";
import { currentUser } from "../../server/auth.js";
import { stripeClient } from "../../server/stripe.js";
import { MAX_SEATS, MIN_SEATS, createTeam, deleteTeam } from "../../server/teams.js";

const PRODUCT = "gitsum-team";

export async function POST(req) {
  if (missingEnv(["STRIPE_SECRET_KEY", "GITSUM_TEAM_SEAT_PRICE_ID", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"]).length)
    return fail(503, "not_configured", "Gitsum Team isn't switched on for this site yet.");

  const user = await currentUser(req);
  if (!user) return fail(401, "login_required", "Sign in to create a team.");

  let body;
  try {
    body = await readJson(req, 4000);
  } catch (err) {
    return fail(err.status || 400, "bad_request", err.message);
  }
  const name = typeof body.name === "string" ? body.name.trim().replace(/\s+/g, " ") : "";
  const seats = Number(body.seats);
  const invalid = [];
  if (!name || name.length > 80) invalid.push("name (1–80 characters)");
  if (!Number.isInteger(seats) || seats < MIN_SEATS || seats > MAX_SEATS) invalid.push(`seats (${MIN_SEATS}–${MAX_SEATS})`);
  if (invalid.length) return fail(400, "bad_request", `Invalid: ${invalid.join(", ")}`);

  const returnPath =
    typeof body.returnPath === "string" && body.returnPath.startsWith("/") && !body.returnPath.startsWith("//") ? body.returnPath : "/team";
  const back = new URL(returnPath, originOf(req));

  let teamId;
  try {
    teamId = await createTeam({ name, ownerId: user.id, seats });
  } catch (err) {
    console.error("[team/checkout] create team failed", err?.message);
    return fail(503, "capacity", "Couldn't create the team. Try again in a moment.");
  }

  const success = new URL(back);
  success.searchParams.set("checkout", "success");
  success.searchParams.set("team", teamId);
  const cancel = new URL(back);
  cancel.searchParams.set("checkout", "cancelled");

  // No client_reference_id / supabase_user_id on purpose: the Gitsum web app's webhook (same Stripe
  // account) keys Pro subscriptions on those, and must not mistake a team purchase for a Pro one.
  const metadata = { product: PRODUCT, team_id: teamId };
  try {
    const session = await stripeClient().checkout.sessions.create({
      mode: "subscription",
      line_items: [
        {
          price: env("GITSUM_TEAM_SEAT_PRICE_ID"),
          quantity: seats,
          adjustable_quantity: { enabled: true, minimum: MIN_SEATS, maximum: MAX_SEATS },
        },
      ],
      customer_email: user.email || undefined,
      subscription_data: { metadata },
      metadata,
      allow_promotion_codes: true,
      success_url: success.toString(),
      cancel_url: cancel.toString(),
    });
    return json(200, { url: session.url, team_id: teamId });
  } catch (err) {
    console.error("[team/checkout] stripe failed", err?.message);
    await deleteTeam(teamId).catch(() => {});
    return fail(503, "capacity", "Couldn't start checkout. Try again in a moment.");
  }
}
