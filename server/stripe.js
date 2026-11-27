import Stripe from "stripe";
import { env } from "./http.js";

let stripe = null;

export function stripeClient() {
  if (stripe) return stripe;
  const key = env("STRIPE_SECRET_KEY");
  if (!key) return null;
  stripe = new Stripe(key, { maxNetworkRetries: 2 });
  return stripe;
}

/** Test hook. */
export function __setStripe(s) {
  stripe = s;
}

export const toIso = (unix) => (unix ? new Date(unix * 1000).toISOString() : null);

/** Billing fields of a Stripe subscription, in the shape of public.gitsum_teams. */
export function billingFields(subscription, customerId) {
  const item = subscription.items?.data?.[0];
  const periodEnd = subscription.current_period_end ?? item?.current_period_end;
  return {
    stripe_customer_id: customerId || subscription.customer || null,
    stripe_subscription_id: subscription.id,
    status: subscription.status,
    seats: Number(item?.quantity ?? subscription.quantity ?? 0) || 0,
    current_period_end: toIso(periodEnd),
    cancel_at_period_end: !!subscription.cancel_at_period_end,
    updated_at: new Date().toISOString(),
  };
}
