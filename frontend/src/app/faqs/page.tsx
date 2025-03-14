import React from 'react';
import { Accordion, AccordionItem, AccordionTrigger, AccordionContent } from '@/components/ui/accordion'; 
import accordionData from '@/data/accordionData.json';

const FAQsPage: React.FC = () => {
    return (
      <div className="h-screen w-full dark:bg-black bg-white  dark:bg-dot-white/[0.4] bg-dot-black/[0.2] relative flex items-center justify-center">
        <div className="py-3 px-7 flex flex-col w-9/12 items-start justify-end min-h-screen dark:bg-black text-white">
          <Accordion type="single" collapsible className="w-full">
            {accordionData.map((item) => (
              <AccordionItem key={item.value} value={item.value}>
                <AccordionTrigger>{item.trigger}</AccordionTrigger>
                <AccordionContent>{item.content}</AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </div>
      </div>
    );
};

export default FAQsPage;