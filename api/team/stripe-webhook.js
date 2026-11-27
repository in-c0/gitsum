// POST /api/team/stripe-webhook — keeps public.gitsum_teams in step with Stripe.
// Events: checkout.session.completed, customer.subscription.created|updated|deleted
// Only objects with metadata.product = "gitsum-team" are handled; everything else is acknowledged and ignored.
// Idempotent: every event writes the subscription's current state, so retries and replays are safe.

import { env, fail, json } from "../../server/http.js";
import { billingFields, stripeClient } from "../../server/stripe.js";
import { saveTeamBilling, teamBySubscription } from "../../server/teams.js";

const PRODUCT = "gitsum-team";
const MAX_PAYLOAD = 1_000_000;

export async function POST(req) {
  const stripe = stripeClient();
  const secret = env("STRIPE_TEAM_WEBHOOK_SECRET") || env("STRIPE_WEBHOOK_SECRET");
  if (!stripe || !secret) return fail(503, "not_configured", "Stripe webhook isn't configured.");

  const signature = req.headers.get("stripe-signature");
  const payload = await req.text();
  if (payload.length > MAX_PAYLOAD) return fail(413, "bad_request", "Payload too large");
  let event;
  try {
    event = await stripe.webhooks.constructEventAsync(payload, signature, secret);
  } catch (err) {
    return fail(400, "bad_signature", `Webhook signature check failed: ${err.message}`);
  }

  try {
    if (event.type === "checkout.session.completed") {
      const session = event.data.object;
      if (session.metadata?.product !== PRODUCT) return json(200, { ignored: "other product" });
      if (session.mode !== "subscription" || !session.subscription) return json(200, { ignored: true });
      const teamId = session.metadata?.team_id;
      if (!teamId) return json(200, { ignored: "no team" });
      const subscription = await stripe.subscriptions.retrieve(session.subscription);
      const saved = await saveTeamBilling(teamId, billingFields(subscription, session.customer));
      if (!saved) return json(200, { ignored: "unknown team" });
    } else if (event.type.startsWith("customer.subscription.")) {
      const subscription = event.data.object;
      if (subscription.metadata?.product !== PRODUCT) return json(200, { ignored: "other product" });
      const teamId = subscription.metadata?.team_id || (await teamBySubscription(subscription.id))?.id;
      if (!teamId) return json(200, { ignored: "unknown subscription" });
      const saved = await saveTeamBilling(teamId, billingFields(subscription, subscription.customer));
      if (!saved) return json(200, { ignored: "unknown team" });
    } else {
      return json(200, { ignored: event.type });
    }
  } catch (err) {
    console.error("[team/stripe-webhook]", event.type, err?.message);
    // 500 makes Stripe retry, which is what we want for a transient database error.
    return fail(500, "save_failed", "Couldn't save the team subscription.");
  }

  return json(200, { received: true });
}
