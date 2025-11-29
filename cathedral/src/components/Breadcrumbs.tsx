import { useParams } from "react-router-dom"
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"

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
    <Breadcrumb>
      <BreadcrumbList>
        {breadcrumbItems.map((item, index) => (
          <BreadcrumbItem key={index}>
            {index < breadcrumbItems.length - 1 ? (
              <BreadcrumbLink href={item.href}>{item.label}</BreadcrumbLink>
            ) : (
              <BreadcrumbPage>{item.label}</BreadcrumbPage>
            )}
            {index < breadcrumbItems.length - 1 && <BreadcrumbSeparator />}
          </BreadcrumbItem>
        ))}
      </BreadcrumbList>
    </Breadcrumb>
  )
}
