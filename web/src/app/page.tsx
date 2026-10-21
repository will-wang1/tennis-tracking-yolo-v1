import { redirect } from "next/navigation";
import { requireUser } from "@/lib/auth";

export const dynamic = "force-dynamic";

export default async function Home() {
  const user = await requireUser();
  redirect(user.role === "head_coach" ? "/coaches" : user.coach_id ? `/coaches/${user.coach_id}` : "/sessions");
}
