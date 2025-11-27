import { useParams } from "react-router-dom"
import { Breadcrumb } from "./ui/breadcrumb";

export function Breadcrumbs() {
  const { "*": path } = useParams();

  let breadcrumbItems = []

  breadcrumbItems.push({ label: "root", href: `/` });

  if (path) {
    path.split("/").reduce((acc, part) => {
      const href = acc ? `${acc}/${part}` : part;
      breadcrumbItems.push({ label: part || "home", href: `/${href}` });
      return href;
    }, "");
  }

  return (
    <div className="font-mono text-sm">
      <Breadcrumb items={breadcrumbItems} />
    </div>
  )
}
