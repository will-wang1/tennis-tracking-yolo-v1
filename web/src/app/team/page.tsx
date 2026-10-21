import { requireHeadCoach } from "@/lib/auth";
import { listCoaches } from "@/lib/db";
import { listUsers, ROLE_LABEL } from "@/lib/users";
import { AddMember, RemoveMember, ResetPassword } from "@/components/TeamForms";

export const dynamic = "force-dynamic";

export default async function TeamPage() {
  const me = await requireHeadCoach();
  const users = listUsers();
  const linked = new Set(users.map((u) => u.coach_id).filter((x) => x != null));
  const coaches = listCoaches().map((c) => ({ id: c.id, name: c.name, taken: linked.has(c.id) }));
  return (
    <>
      <div className="head">
        <div>
          <p className="eyebrow">Head coach</p>
          <h1>Team</h1>
          <p className="lede">Who can sign in. Coaches see only their own sessions; head coaches see everything and set the benchmarks.</p>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Sessions filed under</th><th /></tr></thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.id}>
                <td><b>{u.name}</b>{u.must_change_password && <span className="muted small"> · not signed in yet</span>}</td>
                <td className="nw">{u.email}</td>
                <td className="nw">{ROLE_LABEL[u.role]}</td>
                <td>{u.coach_name ?? <span className="muted">-</span>}</td>
                <td className="nw">
                  {u.id !== me.id && <>
                    <ResetPassword id={u.id} />{" "}
                    <RemoveMember id={u.id} name={u.name} />
                  </>}
                  {u.id === me.id && <span className="muted small">you</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <section className="section panel" style={{ maxWidth: 720 }}>
        <h2>Add an account</h2>
        <AddMember coaches={coaches} />
      </section>
    </>
  );
}
