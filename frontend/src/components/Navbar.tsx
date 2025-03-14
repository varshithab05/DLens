"use client";
import React, { useState } from "react";
import {Menu, MenuItem} from "./ui/navbar-menu";
import { cn } from "@/lib/utils";
import Link from "next/link";

function Navbar({ className }: { className?: string }) {
    const [active, setActive] = useState<string | null>(null);
    return(
        <div className={cn("fixed top-10 inset-x-0 max-w-2xl mx-auto z-50", className)}>
            <Menu setActive={setActive}>
                <Link href={"/"}>
                    <MenuItem item="Home" setActive={setActive} active={active}></MenuItem>
                </Link>
                <Link href={"/playground"}>
                    <MenuItem item="Playground" setActive={setActive} active={active}></MenuItem>
                </Link>
                <Link href={"/faqs"}>
                    <MenuItem item="FAQs" setActive={setActive} active={active}></MenuItem>
                </Link>
                <Link href={"https://github.com/varshithab05"}>
                    <MenuItem item="Github" setActive={setActive} active={active}></MenuItem>
                </Link>
            </Menu>
        </div>
    )
}
export default Navbar 